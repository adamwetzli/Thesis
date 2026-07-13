import pandas as pd
import numpy as np
import optuna
import os
import joblib
from typing import List, Union, Tuple, Dict
from sklearn.ensemble import RandomForestClassifier

# Project Modules
from utils import (BlockingTimeSeriesSplit, plot_optuna_study, plot_nested_wfv_dashboard, 
                   plot_multiple_financial_distributions, plot_nested_feature_importances,
                   plot_nested_correlation_heatmap, plot_nested_conformal_preds,
                   plot_nested_confusion_matrices,
                   plot_nested_reliability_diagrams, generate_performance_table,
                   generate_trading_hps_table, generate_strategy_comparison_table,
                   generate_rrf_leaderboard, generate_tournament_summary_tables)
from custom_backtester import CustomBacktester, sharpe_ratio_
from forex_model_registry import MODELS
from forex_feature_preprocessing import run_feature_selection_pipeline, preprocess_features

# ==============================================================================
# 1. THE CORE ENGINE (Walk-Forward Meta-Accumulation)
# ==============================================================================

def _execute_walk_forward_accumulation(m1_name: str,
                                       m1_hps: dict,
                                       m2_hps: dict,
                                       data: pd.DataFrame,
                                       n_inner_splits: int,
                                       n_purged: int = 10,
                                       n_embargo: int = 10,
                                       existing_memory: list = None,
                                       selected_features: list = None):
    """
    Applies a K-fold Block-Split on the Outer Training Block and accumulates a lagged Dataset for
    the training of M2 (Meta Model).

    The Core Accumulation Engine of the pipeline. Implements a Double-OOS
    layer to generate statistically unbiased signals for the Meta-Model (M2).

    THE "FIT-PREDICT-APPEND" CYCLE:
    To prevent lookahead bias, this function manages data flow across inner folds:
    1. Primary Model (M1) is trained on the current training fold.
    2. M1 generates signals and probabilities on the validation fold (True OOS for M1).
    3. Meta-Model (M2) is trained ONLY on the mistakes M1 made in previous folds
        (accumulated in the buffer) plus any 'existing_memory' from previous years.
    4. M2 generates confidence scores (probabilities) for the current M1 signals.
       Because M2 never saw the current validation set during its training, this
       ensures a second layer of OOS protection.
    5. The current validation set's features and M1's "empirical mistakes" are then
        added to the buffer for the NEXT fold's M2 training.

    EMPIRICAL META-LABELING:
        Unlike theoretical labeling, this engine judges M1 based on the actual Triple Barrier
        outcome ('y_truth'). If M1's directional bet matches the barrier realization, the
        Meta-Label is 1; otherwise, it is 0. This allows M2 to learn the specific behavioral
        weaknesses of the current M1 architecture.

    GLOBAL MEMORY INTEGRATION:
        If 'existing_memory' is provided, the training set for M2 is augmented with
        historical M1 failure patterns from previous outer folds. This mitigates
        the "cold start" problem in early folds and provides a larger, more
        robust statistical sample for the meta-filter.

    Args:
        m1_name (str): Architecture key from the Model Registry.
        m1_hps (dict): Hyperparameters for the Primary (Side) model.
        m2_hps (dict): Hyperparameters for the Meta (Filter) model.
        data (pd.DataFrame): The current preprocessed block of historical data (Outer Training Fold).
        n_inner_splits (int): Number of blocks to split the data into.
        n_purged (int): Bars removed between training/validation to prevent trade overlap.
        n_embargo (int): Bars removed after validation to prevent data leakage.
        existing_memory (list, optional): List of (features, labels) from previous outer blocks.
        selected_features (list, optional): Subset of feature names to use for M1/M2 training.

    Returns:
        tuple: (fold_results, final_mistakes)
            - fold_results: A list of (sigs_m1, probs_m2, price_data) for the Trading HPO backtester.
                Storing M2 probabilities separately allows the threshold to be tuned as a
                trading hyperparameter without retraining the ML models.
            - final_mistakes: A tuple of (features, labels, probs) containing M1's performance
                across this entire block. The 'probs' are the Out-of-Sample confidence scores
                from M2, used to build the Conformal Calibration set.
    """
    if n_inner_splits < 2:
        return [], (pd.DataFrame(), pd.Series(), pd.Series())
    
    # Get M1 Model Config and prepare BTS
    m1_def = MODELS[m1_name]
    cv = BlockingTimeSeriesSplit(n_splits=n_inner_splits, n_purged=n_purged, n_embargo=n_embargo)
    
    # Extract Features/Labels/Meta_Cols
    y_side = data['y_side'].copy()
    y_truth = data['y_truth'].copy()
    meta_cols = ['raw_atr', 'tx_high', 'tx_low', 'tx_close']
    X = data.drop(columns=['y_side', 'y_truth', 'pair'] + meta_cols, errors='ignore')

    if selected_features:
        X = X[selected_features]
    
    # Initialize Buffers for multi-fold memory of features/labels/probabilities
    # Initialize Fold results list (stores M1 signals, M2 probs and ['pair', 'tx_high', 'tx_low', 'tx_close', 'raw_atr'] cols)
    meta_feat_buf, meta_lab_buf, meta_prob_buf = [], [], [] 
    fold_results = []

    for i, (i_train_idx, i_val_idx) in enumerate(cv.split(data)):
        # Get Fold specific Train/Val Data
        X_tr, y_s_tr = X.loc[i_train_idx], y_side.loc[i_train_idx]
        X_val = X.loc[i_val_idx]
        y_t_val = y_truth.loc[i_val_idx]
        
        # Train M1 Model via its model-agnostic bridge
        m1_tmp = m1_def['bridge'](m1_def['class'], m1_hps, X_tr, y_s_tr, n_purged=n_purged, n_embargo=n_embargo)
        # Predict Signal + respective Probability on Fold's Val data
        sigs_m1 = m1_def['predictor'](m1_tmp, X_val)
        probs_m1 = m1_def['prob_predictor'](m1_tmp, X_val)
        
        # Build Meta Model (M2) features
        X_v_m = X_val.copy()
        X_v_m['m1_sig'] = sigs_m1
        X_v_m['m1_prob'] = probs_m1

        if i == 0:
            # Use np.nan for the first fold where M2 hasn't trained yet (doesn't have data yet).
            # This prevents dummy probabilities from polluting the calibration distribution
            # while maintaining the length of the stored lists in the memory (otherwise mismatch of indices).
            meta_prob_buf.append(pd.Series([np.nan] * len(X_v_m), index=X_v_m.index))

        elif i > 0:
            # Build the historical Meta dataset of the data provided in the function
            # (in nested wfv this is built from the inner folds inside an outer fold)
            all_m2_feats = pd.concat(meta_feat_buf)
            all_m2_labs = pd.concat(meta_lab_buf)

            # Existing memory comes from previous Outer Folds (only in nested wfv)
            if existing_memory:
                # Extract previous outer folds' historical datasets
                past_feats = pd.concat([m[0] for m in existing_memory])
                past_labs = pd.concat([m[1] for m in existing_memory])

                # Build the full historical meta dataset across old outer folds and these inner folds
                all_m2_feats = pd.concat([past_feats, all_m2_feats]).fillna(0)
                all_m2_labs = pd.concat([past_labs, all_m2_labs])
            
            # Train the M2 Model
            m2_tmp = RandomForestClassifier(**m2_hps).fit(all_m2_feats, all_m2_labs)

            # Generate probabilities instead of binary signals for tunable filtering (Conformal Q)
            # Alignment Fix: Ensure prediction features match the training schema (union of features in memory)
            # probs_m2 = m2_tmp.predict_proba(X_v_m)[:, 1] # BEFORE (Not dynamic enough for feature selection process in wfv functions)
            X_v_m_predict = X_v_m.reindex(columns=all_m2_feats.columns, fill_value=0)
            probs_m2 = m2_tmp.predict_proba(X_v_m_predict)[:, 1]

            # Accumulate M1 signals and M2 confidence scores for the inner HPO loop
            val_data = data.loc[i_val_idx]
            fold_results.append((pd.Series(sigs_m1, index=val_data.index),
                                 pd.Series(probs_m2, index=val_data.index),
                                 val_data[['pair', 'tx_high', 'tx_low', 'tx_close', 'raw_atr']]))
            
            # Store OOS probabilities for the next fold's calibration
            meta_prob_buf.append(pd.Series(probs_m2, index=X_v_m.index))
        
        # Extract this Fold's M2 labels by comparing M1 labels to MLP TBM (Truth) Labels
        current_y_meta = (sigs_m1 == y_t_val).astype(int)

        # Append new Meta features and M2 labels (M2 probs are handled in the if statement above)
        meta_feat_buf.append(X_v_m)
        meta_lab_buf.append(current_y_meta)

    return fold_results, (pd.concat(meta_feat_buf), pd.concat(meta_lab_buf), pd.concat(meta_prob_buf))


def get_clean_hps(params: dict, prefix: str) -> dict:
    """
    Extracts hyperparameters from an Optuna params dict that start with a specific 
    prefix (e.g. 'm1__') and removes that prefix.
    """
    return {k.replace(prefix, ''): v for k, v in params.items() if k.startswith(prefix)}


def compute_conformal_threshold(calibration_probs, significance=0.1):
    """
    Compute threshold ONCE. O(n log n) for sorting, then O(1) for test time.
    
    Args:
        calibration_probs: Array of M2 probabilities when M1 was wrong
        significance: α level (e.g., 0.1 for 90% confidence)
    
    Returns:
        threshold: The (1-α) quantile value
    """
    if len(calibration_probs) == 0:
        return 0.5  # Default fallback
    
    # Sort once (O(n log n))
    sorted_probs = np.sort(calibration_probs)
    
    # Find the (1-α) quantile index
    n = len(sorted_probs)
    quantile_idx = int(np.ceil((1 - significance) * (n + 1))) - 1
    quantile_idx = max(0, min(quantile_idx, n - 1))
    
    threshold = sorted_probs[quantile_idx]
    
    return threshold


def _rescore_mistakes_via_model(feats, labs, model):
    """
    Re-generates M2 confidence scores for a stored (features, labels) mistake
    tuple using a *live* model, instead of trusting whatever probabilities were
    frozen in at the time the tuple was created.

    This exists because M2 is refit from scratch at several points in this
    pipeline (once per outer fold in run_nested_wfv, and even multiple times
    within a single fold's inner accumulation loop). Conformal calibration is
    only statistically valid when the calibration scores and the live test-time
    scores come from the SAME scoring function. Reusing old, frozen probabilities
    generated by a now-discarded model instance compares apples to oranges: if
    the new model's output scale/distribution has drifted at all from the old
    one's, the resulting p-values can be systematically pushed toward 0 or 1,
    silently gating almost all (or none) of a pair's signals for no real reason.

    Args:
        feats (pd.DataFrame): Stored feature rows (from a mistake-tuple's [0]).
        labs (pd.Series): Stored M1-correctness labels (0 = M1 was wrong), aligned to feats.
        model: A fitted sklearn-compatible classifier with .predict_proba and
            .feature_names_in_ (i.e. an M2 instance).

    Returns:
        list[float]: Rescored P(M1 correct) probabilities for the rows where M1 was wrong
            (label == 0), which is the population the conformal calibration set is built from.
    """
    mistake_mask = (labs == 0)
    if hasattr(mistake_mask, 'values'):
        mistake_mask = mistake_mask.values
    if not np.any(mistake_mask):
        return []

    feats_mistakes = feats.loc[mistake_mask] if hasattr(feats, 'loc') else feats[mistake_mask]
    feats_aligned = feats_mistakes.reindex(columns=model.feature_names_in_, fill_value=0)
    return model.predict_proba(feats_aligned)[:, 1].tolist()


def evaluate_on_test_set(m1_model, m1_name, m2_model, test_data, t_hps, 
                         calibration_probs=None, selected_features=None,
                         fitted_preprocessor=None, return_details=False,
                         num_trials=None, variance_of_sharpes=None, meta_hpo_psrs=None,
                         initial_cash=10_000, tc_per_unit=0.0001, slippage_per_unit=0.0002,
                         min_qty=1, max_qty=100_000) -> Union[float, Tuple[float, Dict]]:
    """
    Evaluates a trained ML pipeline (M1 + M2) on an unseen test block using Conformal
    signal filtering and the institutional backtester.
    """
    # 1. Preprocessing & Leakage Prevention
    meta_cols = ['raw_atr', 'tx_high', 'tx_low', 'tx_close', 'num_atr']
    y_truth = test_data['y_truth'].copy()
    X_test_raw = test_data.drop(columns=['y_side', 'y_truth', 'pair'] + meta_cols, errors='ignore')
    
    if fitted_preprocessor:
        X_test_pre, _ = preprocess_features(X_test_raw, fitted_preprocessor=fitted_preprocessor)
    else:
        print('[CRITICAL MISTAKE] No preprocessor found! Cannot preprocess OOS Test Data!')
        print('Continuing with unprocessed test data!')
        X_test_pre = X_test_raw
        
    if selected_features:
        X_test_pre = X_test_pre[selected_features]

    # 2. Base Predictions
    # Calculate M2 probabilities and M1 signals
    side_m1 = MODELS[m1_name]['predictor'](m1_model, X_test_pre)
    prob_m1 = MODELS[m1_name]['prob_predictor'](m1_model, X_test_pre)
    
    # Build M2 Model feature set
    X_meta = X_test_pre.copy()
    X_meta['m1_sig'] = side_m1
    X_meta['m1_prob'] = prob_m1
    
    # Alignment Fix: Ensure prediction features match M2's training schema
    # probs_m2 = m2_model.predict_proba(X_meta)[:, 1] # BEFORE
    X_meta_predict = X_meta.reindex(columns=m2_model.feature_names_in_, fill_value=0)
    probs_m2 = m2_model.predict_proba(X_meta_predict)[:, 1]

    # 3. Backtesting Loop (pairwise)
    pair_results = []
    benchmark_results = []
    details = {}
    
    # Track thresholds for plotting average
    pair_significances = []
    pair_thresholds = []
    
    # Pre-calculate global p-values if calibration data exists
    # Conformal Filtering
    # Null Hypothesis (H₀): The current M1 signal is a mistake (M2 will predict low confidence).
    if calibration_probs is not None and len(calibration_probs) > 0:
        n = len(calibration_probs)
        # Broadcasts cal array and p_m2 col vector to generate (n, len(p_m2)) shape, then sums along rows
        # "Given the distribution of M2's confidence scores on past mistakes, how unusual is this current M2 confidence?"
        all_p_vals = (np.sum(calibration_probs >= probs_m2[:, None], axis=1) + 1) / (n + 1) # Prob that cal_array >= a p_m2 pred
        all_conf = 1.0 - all_p_vals
    else:
        all_p_vals = None
        all_conf = None

    # Ensure ATR is available under consistent name
    atr_series = test_data['raw_atr'] if 'raw_atr' in test_data.columns else test_data['num_atr']

    for pair in test_data['pair'].unique():
        pair_mask = test_data['pair'] == pair
        test_pair = test_data[pair_mask]
        
        # Get Per-Pair Significance
        p_sig = t_hps.get(f'{pair}_meta_significance', 0.1)
        pair_significances.append(p_sig)
        
        # Calculate Per-Pair Conformal Threshold for plotting info
        p_thresh = compute_conformal_threshold(calibration_probs, p_sig)
        pair_thresholds.append(p_thresh)

        # Prepare Signal & Confidence for THIS pair
        if all_p_vals is not None:
            p_vals_pair = all_p_vals[pair_mask]
            c_pair = pd.Series(all_conf[pair_mask], index=test_pair.index)
            # Final descision
            s_pair = pd.Series(side_m1[pair_mask] * (p_vals_pair <= p_sig).astype(int), index=test_pair.index)
        else:
            c_pair = None
            s_pair = pd.Series(side_m1[pair_mask] * (probs_m2[pair_mask] >= 0.5).astype(int), index=test_pair.index)

        # Prepare Price Data
        p_bt = test_pair[['tx_high', 'tx_low', 'tx_close']].copy()
        p_bt.columns = ['high', 'low', 'close']
        
        # Calculate dynamic barriers using per-pair HPs
        atr_pair = atr_series[pair_mask]
        sl_series = atr_pair * t_hps[f'{pair}_sl_mult']
        tp_series = atr_pair * t_hps[f'{pair}_tp_mult']
        max_notional_exposure_pct = t_hps.get(f'{pair}_max_notional_exposure_pct', 100.0) # Retrieve from t_hps or use default
        
        bt = CustomBacktester(ohlc=p_bt,
                              signals=s_pair,
                              initial_cash=initial_cash,
                              tc_per_unit=tc_per_unit,
                              slippage_per_unit=slippage_per_unit,
                              max_notional_exposure_pct=max_notional_exposure_pct)
        _ = bt.run(sl=sl_series,
                   tp=tp_series,
                   max_holding_periods=t_hps[f'{pair}_mh'],
                   risk_pct=t_hps[f'{pair}_risk_pct'],
                   conformal_confidence=c_pair,
                   significance=p_sig,
                   kelly_fraction=t_hps[f'{pair}_kelly_fraction'],
                   min_qty=min_qty,
                   max_qty=max_qty,
                   is_distance=True)
        
        # Convert PSR to p_value proxy where H0: The true Sharpe Ratio is less than or equal to the benchmark (SR=0 here)
        # Pass to get_stats method so that pDFR can be computed
        meta_hpo_psrs_arr = np.array(meta_hpo_psrs)
        p_values = 1 - meta_hpo_psrs_arr

        # Build tuple of correct predictions by M1 and predicted probabilities of M2 and pass it to get_stats for brier score
        m1_correct = (side_m1 == y_truth).astype(int)
        brier_tuple = (m1_correct, probs_m2)

        # Fetch stats with DSR metadata
        stats = bt.get_stats(
            num_trials=num_trials, 
            variance_of_sharpes=variance_of_sharpes, 
            p_values=p_values,
            brier_tuple=brier_tuple)
        
        # Preserve granular data
        stats_clean = {k: v for k, v in stats.items() if k not in ['portfolio_df', 'trade_history']}
        stats_clean['pair'] = pair
        pair_results.append(stats_clean)

        # --- BENCHMARK STRATEGIES ---
        # 1. Buy & Hold (Mathematical Passive Equity)
        # Calculated as: Buy at t=0, Hold until end. No trades/costs.
        bh_equity = initial_cash * (p_bt['close'] / p_bt['close'].iloc[0])
        bh_sharpe = sharpe_ratio_(bh_equity)
        
        # 2. SMA Crossover (50/200 Pure Logic)
        # Uses Backtester with infinite SL/TP/Timeout to force pure crossover logic.
        fast_ma = p_bt['close'].rolling(50).mean()
        slow_ma = p_bt['close'].rolling(200).mean()
        sma_sigs = np.where(fast_ma > slow_ma, 1, -1)
        sma_sigs[np.isnan(fast_ma) | np.isnan(slow_ma)] = 0
        bt_sma = CustomBacktester(ohlc=p_bt, signals=pd.Series(sma_sigs, index=test_pair.index),
                                  initial_cash=initial_cash, tc_per_unit=tc_per_unit, 
                                  slippage_per_unit=slippage_per_unit)
        stats_sma = bt_sma.run(sl=9999.0, tp=9999.0, max_holding_periods=len(p_bt), is_distance=True)
        
        # 3. M1 Only (Raw Directional Signals + Optimized Guardrails)
        # Uses M1 signals but same SL/TP logic as complex model (No Meta filter).
        bt_simple = CustomBacktester(ohlc=p_bt, signals=pd.Series(side_m1[pair_mask], index=test_pair.index),
                                     initial_cash=initial_cash, tc_per_unit=tc_per_unit, 
                                     slippage_per_unit=slippage_per_unit,
                                     max_notional_exposure_pct=max_notional_exposure_pct)
        stats_simple = bt_simple.run(sl=sl_series, tp=tp_series, max_holding_periods=t_hps[f'{pair}_mh'],
                                     risk_pct=t_hps[f'{pair}_risk_pct'], 
                                     kelly_fraction=t_hps[f'{pair}_kelly_fraction'],
                                     is_distance=True)

        # 4. M1 + M2 (Fixed)
        # Uses M1 signals filtered by M2 with a fixed 0.5 threshold (No Conformal Calibration).
        s_fixed = pd.Series(side_m1[pair_mask] * (probs_m2[pair_mask] >= 0.5).astype(int), index=test_pair.index)
        bt_fixed = CustomBacktester(ohlc=p_bt, signals=s_fixed,
                                    initial_cash=initial_cash, tc_per_unit=tc_per_unit, 
                                    slippage_per_unit=slippage_per_unit,
                                    max_notional_exposure_pct=max_notional_exposure_pct)
        stats_fixed = bt_fixed.run(sl=sl_series, tp=tp_series, max_holding_periods=t_hps[f'{pair}_mh'],
                                   risk_pct=t_hps[f'{pair}_risk_pct'], 
                                   kelly_fraction=t_hps[f'{pair}_kelly_fraction'],
                                   is_distance=True)

        # 5. M1 + M2 (Global Baseline)
        # Uses median trading hyperparameters across all pairs in this fold to prove "Local Adaptation" efficacy.
        # This is a one-size-fits-all approach using the Global Brain but Global (Median) Gear.
        median_sl_mult = np.median([t_hps[k] for k in t_hps if '_sl_mult' in k])
        median_tp_mult = np.median([t_hps[k] for k in t_hps if '_tp_mult' in k])
        median_mh = int(np.median([t_hps[k] for k in t_hps if '_mh' in k]))
        median_sig = np.median([t_hps[k] for k in t_hps if '_meta_significance' in k])
        
        sl_global = atr_pair * median_sl_mult
        tp_global = atr_pair * median_tp_mult
        
        if all_p_vals is not None:
            s_global = pd.Series(side_m1[pair_mask] * (all_p_vals[pair_mask] <= median_sig).astype(int), index=test_pair.index)
        else:
            s_global = s_fixed # Fallback if no conformal
            
        bt_global = CustomBacktester(ohlc=p_bt, signals=s_global,
                                     initial_cash=initial_cash, tc_per_unit=tc_per_unit, 
                                     slippage_per_unit=slippage_per_unit,
                                     max_notional_exposure_pct=max_notional_exposure_pct)
        stats_global = bt_global.run(sl=sl_global, tp=tp_global, max_holding_periods=median_mh,
                                     risk_pct=t_hps[f'{pair}_risk_pct'], # Keep risk % per pair as it's capital management
                                     kelly_fraction=t_hps[f'{pair}_kelly_fraction'],
                                     is_distance=True)

        benchmark_results.append({
            'pair': pair,
            'Buy and Hold': bh_sharpe,
            'SMA Crossover': stats_sma.get('sharpe', 0.0),
            'M1 Only': stats_simple.get('sharpe', 0.0),
            'M1 + M2 (Fixed)': stats_fixed.get('sharpe', 0.0),
            'M1 + M2 (Global)': stats_global.get('sharpe', 0.0),
            'M1 + M2 (Conformal)': stats.get('sharpe', 0.0)
        })

        if return_details:
            details[pair] = {
                'ohlc': p_bt,
                'trade_history': stats.get('trade_history', []),
                'equity_history': stats.get('portfolio_df', pd.DataFrame()),
                'stats': stats,
                'initial_cash': initial_cash
            }

    # Construct Granular DataFrames
    granular_df = pd.DataFrame(pair_results).set_index('pair')
    benchmark_df = pd.DataFrame(benchmark_results).set_index('pair')

    # Use side_m1 (raw signals) for plotting to show all predictions, even those filtered out
    # Use means for plotting compatibility
    avg_sig = np.mean(pair_significances) if pair_significances else 0.1
    avg_thresh = np.mean(pair_thresholds) if pair_thresholds else 0.5
    conf_preds_tuple = (side_m1, y_truth, probs_m2, avg_thresh, avg_sig)

    if return_details:
        return granular_df, benchmark_df, details, conf_preds_tuple
    
    return granular_df, benchmark_df, conf_preds_tuple


# ==============================================================================
# 2. THE OPTIMIZATION ENGINE (Nested Tuning)
# ==============================================================================

def optimize_pipeline(m1_name: str,
                      tuning_data: pd.DataFrame,
                      n_inner_splits,
                      n_purged,
                      n_embargo,
                      opt_metric='sharpe',
                      optim_dir='tournament',
                      n_model_trials=20,
                      n_trading_trials=50,
                      initial_cash=10_000,
                      min_qty=1,
                      max_qty=100_000,
                      tc_per_unit=0.0001,
                      slippage_per_unit=0.0002,
                      existing_memory: List = None,
                      selected_features: list = None,
                      enable_plotting=False):
    """
    Performs a nested hyperparameter optimization (HPO) for both the ML models and
    the trading execution parameters.

    THE TWIN-ENGINE OPTIMIZATION:
    1. Model HPO (Outer Optuna Study): Finds the optimal architectural parameters for
        the primary model (M1) and the secondary meta-filter (M2). It uses
        '_execute_walk_forward_accumulation' to generate realistic Out-of-Sample (OOS)
        signals and confidence scores for every trial.
    2. Trading HPO (Inner Optuna Study): For every architectural trial, a second study
        optimizes the risk management parameters (ATR-based SL/TP multipliers, Risk Parity %, 
        Kelly Scaling, Max Hold) AND the Conformal significance level using the 'CustomBacktester'.

    Args:
        m1_name (str): The name of the primary architecture (e.g., 'XGBoost', 'LSTM').
        tuning_data (pd.DataFrame): The preprocessed training block for this optimization cycle.
        n_inner_splits (int): Number of folds for the inner walk-forward process.
        n_purged (int): Overlap removal between training and validation.
        n_embargo (int): Gap removal after validation to prevent lookahead.
        opt_metric (str): The performance goal ('sharpe', 'return', 'mdd', 'calmar', 'dsr').
        n_model_trials (int): Number of architectural HPO iterations.
        n_trading_trials (int): Number of trading parameter iterations per architecture trial.
        existing_memory (List): TBM/M1 performance history (features, labels, probs) from previous blocks.
        selected_features (list): Subset of features determined by the FS pipeline.
        enable_plotting (bool): If True, generates Optuna visualization plots (Production phase only).

    Returns:
        tuple: (m1_final, m2_final, best_m1_hps, best_trading_hps, final_mistakes, hpo_metadata)
                Includes the fitted models, the winning parameters, the new
                set of M1 mistakes, and metadata (n_trials, variance) for DSR.
    """
    m1_def = MODELS[m1_name]

    def model_objective(trial):
        # 1. Suggested HPs are namespaced (e.g. 'm1__n_estimators' and 'm2__max_depth')
        m1_hps_raw = m1_def['suggest'](trial)
        m2_depth = trial.suggest_int('m2__max_depth', 3, 10)
        
        # 2. Clean prefixes before passing to the models
        m1_hps_clean = get_clean_hps(m1_hps_raw, 'm1__')
        # Fallback if the registry didn't use prefixes yet (robustness)
        if not m1_hps_clean: m1_hps_clean = m1_hps_raw
        
        # M2 HPs are fixed and minimal
        m2_hps_clean = {'n_estimators': 100, 'max_depth': m2_depth, 'random_state': 42}
        
        fold_results, fold_memory = _execute_walk_forward_accumulation(m1_name=m1_name,
                                                                       m1_hps=m1_hps_clean,
                                                                       m2_hps=m2_hps_clean,
                                                                       data=tuning_data,
                                                                       n_inner_splits=n_inner_splits,
                                                                       n_purged=n_purged,
                                                                       n_embargo=n_embargo,
                                                                       existing_memory=existing_memory,
                                                                       selected_features=selected_features)
        
        # Extract outcomes and OOS probabilities for building the calibration set
        # Meta Feature Buffer, Meta Label Buffer, Meta Probability Buffer
        _, f_labs, f_probs = fold_memory # (Only from Tuning Data provided in the above wfv acc. function)

        def trading_objective(t_trial):
            # Extract unique pairs from tuning data
            unique_pairs = tuning_data['pair'].unique()
            
            # Suggest per-pair HPs
            pair_hps = {}
            for pair in unique_pairs:
                pair_hps[pair] = {
                    'sl_mult': t_trial.suggest_float(f'{pair}_sl_mult', 1.0, 5.0),
                    'tp_mult': t_trial.suggest_float(f'{pair}_tp_mult', 1.0, 5.0),
                    'risk_pct': t_trial.suggest_float(f'{pair}_risk_pct', 0.005, 0.03),
                    'kelly_fraction': t_trial.suggest_float(f'{pair}_kelly_fraction', 0.1, 1.0),
                    'mh': t_trial.suggest_int(f'{pair}_mh', 5, 50),
                    'max_notional_exposure_pct': t_trial.suggest_float(f'{pair}_max_notional_exposure_pct', 50.0, 100.0, step=1.0),
                    'meta_significance': t_trial.suggest_float(f'{pair}_meta_significance', 0.01, 0.2)
                }

            # Build the base calibration set (probabilities assigned to past mistakes)
            # It is the empirical distribution of "M2's confidence when M1 fails."
            base_cal = []
            if existing_memory:
                for _, past_labs, past_probs in existing_memory:
                    # Filter out NaNs (from first folds) to prevent pollution
                    valid_mask = ~np.isnan(past_probs)
                    base_cal.extend(past_probs[valid_mask & (past_labs == 0)].tolist()) # label 0 -> M1 was incorrect (bad trade)
            
            # Combine past mistakes with current inner-fold mistakes (all OOS relative to M2)
            # Filter out NaNs (from first folds) to prevent pollution
            curr_valid_mask = ~np.isnan(f_probs)
            current_mistakes = f_probs[curr_valid_mask & (f_labs == 0)].tolist() # label 0 -> M1 was incorrect (bad trade)
            cal_arr = np.array(base_cal + current_mistakes)

            # Initialize Fold Metrics
            fold_metrics = []
            # Initialize dictionary to collect stats across folds (Metadata)
            fold_stats = {
                'total_return': [], 'sharpe': [], 'probabilistic_sharpe': [],
                'max_dd': [], 'cagr': [], 'win_rate': [], 'profit_factor': [],
                'n_trades': [], 'avg_capital_exposure': [], 'avg_trade_size': []
            }
            
            # M1 Pred. Label, M2 Pred. Prob., Prices/ATR/Pair
            for s_m1, p_m2, p in fold_results:
                pair_metrics = []
                fold_pair_stats = {k: [] for k in fold_stats.keys()} # Initialize dictionary to collect stats across pairs for this fold (Metadata)
                
                # Pairwise Backtest on Validation Folds
                for pair in p['pair'].unique():
                    pair_mask = p['pair'] == pair
                    p_hps = pair_hps[pair] # Get this pair's specific HPs
                    
                    # Conformal Confidence for THIS pair
                    # Null Hypothesis (H₀): The current M1 signal is a mistake (M2 will predict low confidence).
                    if len(cal_arr) > 0:
                        n = len(cal_arr)
                        # Use THIS pair's specific significance threshold
                        # Broadcasts cal array and p_m2 col vector to generate (n, len(p_m2)) shape, then sums along rows
                    # "Given the distribution of M2's confidence scores on past mistakes, how unusual is this current M2 confidence?"
                        p_vals = (np.sum(cal_arr >= p_m2[pair_mask].values[:, None], axis=1) + 1) / (n + 1) # Prob that cal_array >= a p_m2 pred
                        p_vals_series = pd.Series(p_vals, index=s_m1[pair_mask].index)
                        conf_series = 1.0 - p_vals_series
                        s_pair = s_m1[pair_mask] * (p_vals_series <= p_hps['meta_significance']).astype(int) # Final Decision
                    else:
                        conf_series = None
                        s_pair = s_m1[pair_mask] * (p_m2[pair_mask] >= 0.5).astype(int)

                    # Get pair ohlc and rename
                    p_pair = p[pair_mask][['tx_high', 'tx_low', 'tx_close', 'raw_atr']]
                    p_pair.columns = ['high', 'low', 'close', 'raw_atr']
                    
                    # Symmetric Volatility Barriers using THIS pair's mults
                    sl_series = p_pair['raw_atr'] * p_hps['sl_mult']
                    tp_series = p_pair['raw_atr'] * p_hps['tp_mult']

                    # Initialize Backtester
                    bt = CustomBacktester(ohlc=p_pair[['high', 'low', 'close']],
                                          signals=s_pair,
                                          initial_cash=initial_cash,
                                          tc_per_unit=tc_per_unit,
                                          slippage_per_unit=slippage_per_unit,
                                          max_notional_exposure_pct=p_hps['max_notional_exposure_pct'])
                    
                    # Run the Backtest with HPs chosen for THIS pair
                    # stats -> {'financial statistic' : value} i.e. {'sharpe' : 1.0, 'win_rate' : 0.46}
                    stats = bt.run(sl=sl_series,
                                   tp=tp_series,
                                   max_holding_periods=p_hps['mh'],
                                   risk_pct=p_hps['risk_pct'],
                                   conformal_confidence=conf_series,
                                   significance=p_hps['meta_significance'],
                                   kelly_fraction=p_hps['kelly_fraction'],
                                   min_qty=min_qty,
                                   max_qty=max_qty,
                                   is_distance=True)
                    
                    if not stats: continue
                    
                    if opt_metric == 'return': raw_score = stats['total_return']
                    elif opt_metric == 'sharpe': raw_score = stats['sharpe']
                    elif opt_metric == 'mdd': raw_score = stats['max_dd']
                    elif opt_metric == 'calmar': raw_score = stats['total_return'] / abs(stats['max_dd']) if stats['max_dd'] != 0 else stats['total_return']
                    else: raw_score = stats['total_return']
                    
                    # In any case, collect the score that is to be optimized
                    pair_metrics.append(raw_score)

                    # Collect all stats for this pair (Metadata)
                    for k in fold_stats.keys():
                        fold_pair_stats[k].append(stats.get(k, 0))

                if pair_metrics:
                    # Accumulate performance across all pairs for this fold's validation data
                    fold_metrics.append(np.mean(pair_metrics))

                # Accumulate stats across all pairs for this fold (Metadata)
                for k in fold_stats.keys():
                    if fold_pair_stats[k]:
                        fold_stats[k].append(np.mean(fold_pair_stats[k]))

            # --- After all folds are done ---

            # Calculate trial mean for each stat across folds (Metadata)
            trial_means = {k: np.mean(v) if v else 0.0 for k, v in fold_stats.items()}
            # Store trial means in the inner study
            all_inner_stats = t_trial.study.user_attrs.get('all_inner_stats', {k: [] for k in fold_stats.keys()})
            for k, v in trial_means.items():
                all_inner_stats[k].append(v)
            t_trial.study.set_user_attr('all_inner_stats', all_inner_stats)
            
            return np.mean(fold_metrics) if fold_metrics else -10.0

        t_study = optuna.create_study(direction='maximize')
        t_study.optimize(trading_objective, n_trials=n_trading_trials)
        
        # Save best trading study for visualization if enabled
        if enable_plotting:
            trial.set_user_attr('t_study', t_study)
            
        trial.set_user_attr('best_t', t_study.best_params)
        
        # Retrieve inner, trading study specific, stats so the outer model trial can see them
        all_inner_stats = t_study.user_attrs.get('all_inner_stats', {})
        
        # Store in outer study
        all_stats = trial.study.user_attrs.get('all_stats', {k: [] for k in all_inner_stats.keys()})
        for k, v in all_inner_stats.items():
            all_stats[k].extend(v)
        trial.study.set_user_attr('all_stats', all_stats)
        
        return t_study.best_value

    m_study = optuna.create_study(direction='maximize')
    m_study.optimize(model_objective, n_trials=n_model_trials)
    
    # Optional Visualization (Phase 2 only)
    if enable_plotting:
        print(f"Generating Optuna Visualization plots for {m1_name}...")
        plot_optuna_study(m_study, m1_name, title_suffix=f"Model_{m1_name}", optim_cat=optim_dir)
        
        # Also plot the best trading study
        best_t_study = m_study.best_trial.user_attrs.get('t_study')
        if best_t_study:
            plot_optuna_study(best_t_study, m1_name, title_suffix=f"Trading_(Best)_{m1_name}", optim_cat=optim_dir)
    
    # Extract Best Params from Study
    best_model_params = m_study.best_params
    best_m1_hps = get_clean_hps(best_model_params, 'm1__')
    # Fallback if no prefix found
    if not best_m1_hps: best_m1_hps = {k:v for k,v in best_model_params.items() if not k.startswith('m2__')}
    
    m2_depth = best_model_params.get('m2__max_depth', 5)
    best_m2_hps = {'n_estimators': 100, 'max_depth': m2_depth, 'random_state': 42}
    
    best_t_hps = m_study.best_trial.user_attrs['best_t']
                               
    # Collect all financial statistics
    all_stats = m_study.user_attrs.get('all_stats', {})

    # fill Metadata with number of optimization trials (n_trading_trials * n_model_trials) as well as V[SR_n] for all n
    hpo_metadata = {'n_trials': len(all_stats['sharpe']),
                    'variance': np.var(all_stats['sharpe']) if len(all_stats['sharpe']) > 1 else 0.0}
    
    # Update hpo_metadata with other collected stats
    # Collecting every single Sharpe Ratio generated during the entire search, as well as n_trials (n_model_trials * n_trading_trials).         
    # and passing this data to the OOS evaluation ensures the most rigorous computation of the deflated Sharpe ratio.
    for k, v in all_stats.items():
        # Simple pluralization (e.g., 'total_return' -> 'total_returns')
        plural_key = k + 's' if not k.endswith('s') else k
        hpo_metadata[plural_key] = v
    
    print(f"Finalizing M1 and M2 models with best parameters...")
    _, final_mistakes = _execute_walk_forward_accumulation(m1_name=m1_name,
                                                           m1_hps=best_m1_hps,
                                                           m2_hps=best_m2_hps,
                                                           data=tuning_data,
                                                           n_inner_splits=n_inner_splits,
                                                           n_purged=n_purged,
                                                           n_embargo=n_embargo,
                                                           existing_memory=existing_memory,
                                                           selected_features=selected_features)
    
    # Build full training set for final M2 Model
    full_m2_feats = final_mistakes[0]
    full_m2_labs = final_mistakes[1]
    if existing_memory:
        full_m2_feats = pd.concat([m[0] for m in existing_memory] + [full_m2_feats])
        full_m2_labs = pd.concat([m[1] for m in existing_memory] + [full_m2_labs])

    # Train final M2 Model
    m2_final = RandomForestClassifier(**best_m2_hps).fit(full_m2_feats, full_m2_labs)
    
    # Build full training set for final M1 Model
    meta_cols = ['raw_atr', 'tx_high', 'tx_low', 'tx_close']
    X_final = tuning_data.drop(columns=['y_side', 'y_truth', 'pair'] + meta_cols, errors='ignore')
    if selected_features:
        X_final = X_final[selected_features]

    # Train final M1 Model
    m1_final = m1_def['bridge'](m1_def['class'], best_m1_hps, X_final, tuning_data['y_side'], n_purged=n_purged, n_embargo=n_embargo)

    # RE-SCORE final_mistakes THROUGH THE ACTUAL DEPLOYED m2_final MODEL.
    # _execute_walk_forward_accumulation generates final_mistakes' probabilities using
    # transient, per-inner-fold `m2_tmp` instances that are thrown away once the inner
    # loop finishes -- NOT m2_final, the model that will actually score live signals.
    # Conformal calibration is only valid when calibration scores and live scores come
    # from the same scoring function, so leaving the stale probabilities in place would
    # make final_mistakes silently miscalibrated relative to the model it's meant to
    # describe. This matters beyond just this fold too: final_mistakes gets carried
    # forward as "memory" for future folds (see run_nested_wfv), so keeping it
    # self-consistent here is what makes that downstream rescoring meaningful.
    final_feats, final_labs, _stale_probs = final_mistakes
    final_feats_aligned = final_feats.reindex(columns=m2_final.feature_names_in_, fill_value=0)
    rescored_probs = pd.Series(m2_final.predict_proba(final_feats_aligned)[:, 1], index=final_feats.index)
    final_mistakes = (final_feats, final_labs, rescored_probs)

    return m1_final, m2_final, best_m1_hps, best_t_hps, final_mistakes, hpo_metadata

# ==============================================================================
# 3. PHASE 1: THE TOURNAMENT (Unbiased Peer Comparison)
# ==============================================================================

def run_nested_wfv(data: pd.DataFrame,
                   model_names: List[str],
                   n_outer_splits,
                   n_inner_splits,
                   n_model_trials,
                   n_trading_trials,
                   opt_metric='sharpe',
                   n_purged=10,
                   n_embargo=10,
                   meta_memory_window=3,
                   mi_thresh=0.005,
                   pi_threshold=0.0,
                   corr_thresh=0.90,
                   pi_model='RF',
                   initial_cash=10_000,
                   tc_per_unit=0.0001,
                   slippage_per_unit=0.0002,
                   min_qty=1,
                   max_qty=100_000):
    """
    Executes a Nested Walk-Forward Optimization tournament to evaluate model architectures.
   
    METHODOLOGY:
    1. Outer Loop (Unbiased Evaluation): Splits the global dataset into K-Blocks. Each outer test block
        serves as an unseen "Future" to test the model's robustness across time.
    2. Shared Preprocessing: Within each outer fold, data is automatically scaled/encoded once based 
        on feature naming conventions (prefixes like 'num_', 'cat_').
    3. Shared Feature Selection: A 3-step selection pipeline (Correlation, MI, Permutation) is run once 
        per outer fold using an RF proxy to ensure a fair and efficient comparison.
    4. Inner Loop (Tournament): Each candidate architecture is optimized using the 'optimize_pipeline'
        engine on the same preprocessed and selected feature set.
    5. Conformal Peer Evaluation: Unlike standard HPO, this tournament evaluates the "Conformalized" 
        version of each model. It uses the sliding window of historical mistakes to calculate
        dynamic p-values for every OOS signal, ensuring the tournament results reflect
        true production-grade risk management.

    Args:
        data (pd.DataFrame): The raw aggregated forex dataset.
        model_names (List[str]): List of architecture names to compete in the tournament.
        n_outer_splits (int): Number of unbiased validation blocks (Outer Folds).
        n_inner_splits (int): Number of cross-validation folds for tuning (Inner Folds).
        n_model_trials (int): Number of architectural HPO trials.
        n_trading_trials (int): Number of trading parameter trials per model trial.
        opt_metric (str): Metric to optimize ('sharpe', 'return', 'mdd', 'calmar').
        n_purged (int): bars removed between sets to prevent trade overlap.
        n_embargo (int): bars removed after validation to prevent data leakage.
        meta_memory_window (int): Sliding window of past fold performance history (features, labels, probs) 
            used to train the Meta-Model and build the Conformal Calibration set.
        mi_thresh (float): Mutual Information score threshold for feature selection.
        pi_threshold (float): Permutation Importance threshold for feature selection.

    Returns:
        dict: A leaderboard ranking each architecture by its mean performance across all outer test periods.
    """
    granular_leaderboard = {m_name: [] for m_name in model_names} # Stores per-pair DataFrames per fold
    model_trading_hps_folds = {m_name: [] for m_name in model_names} # Stores per-pair HPs per fold
    strategy_comparison_leaderboard = {m_name: [] for m_name in model_names} # Stores benchmark comparisons per fold
    model_memories = {m_name: [] for m_name in model_names} # Stores the cross-outer fold model-specific M2 datasets to be as efficient as possible with the data
    full_results = {m_name: [] for m_name in model_names}   # 
    mi_pi_folds = {f"Fold {j+1}" : [] for j in range(n_outer_splits)} # Stores the Mutual Info/Permutation Importance scores of all features across all folds
    Xs_folds = {f"Fold {j+1}" : [] for j in range(n_outer_splits)} # Stores the Features before and after correlation filtering across all folds
    model_conf_preds_folds = {m_name : [] for m_name in model_names} # Stores the model-specific data needed to construct the OOS conformal predictions plot

    outer_cv = BlockingTimeSeriesSplit(n_splits=n_outer_splits, n_purged=n_purged, n_embargo=n_embargo)

    for i, (o_train_idx, o_test_idx) in enumerate(outer_cv.split(data)):
        print(f"\n--- ENTERING OUTER FOLD: {i + 1}/{n_outer_splits} ---")
        
        o_train = data.loc[o_train_idx]
    
        # Separate Labels from features
        labels_train = o_train[['y_side', 'y_truth', 'pair']].copy()
        y_train = o_train['y_side'].copy()

        # Separate Raw Execution Metadata (OHLC/ATR)
        # Needed for backtesting inside pipeline optimization
        meta_cols = ['num_atr', 'tx_high', 'tx_low', 'tx_close']
        execution_metadata_train = o_train[meta_cols].copy().rename(columns={'num_atr': 'raw_atr'})

        # Fit preprocessor ONLY on training raw features
        # Technically do it for every inner fold but it is slightly more computationally expensive
        # and we are not fitting on the outer test data so there is no severe leakage!
        X_raw = o_train.drop(columns=['y_side', 'y_truth', 'pair'] + meta_cols, errors='ignore')
        X_preprocessed, fitted_preprocessor = preprocess_features(X_raw)
        
        # Run Feature Selection once per outer training fold and for ALL models
        # Technically do it for every inner fold but that is too expensive computationally
        X_selected_df, mi_pi_series, Xs_series = run_feature_selection_pipeline(X=X_preprocessed,
                                                                                y=y_train,
                                                                                model=pi_model, # defaults to RandomForest
                                                                                corr_thresh=corr_thresh, 
                                                                                mi_thresh=mi_thresh,
                                                                                pi_thresh=pi_threshold)
        mi_pi_folds[f"Fold {i+1}"] = mi_pi_series # Store Scores in correct fold key
        Xs_folds[f"Fold {i+1}"] = Xs_series # Stores the before/after features in correct fold key
        selected_features = X_selected_df.columns.tolist() 

        # Fresh concatenation of labels, preprocessed features, and execution metadata
        o_train_preprocessed = pd.concat([labels_train, X_preprocessed, execution_metadata_train], axis=1)

        for i, m_name in enumerate(model_names):
            print(f"Evaluating {m_name} [{i+1}/{len(model_names)}]...")
            memory_window = model_memories[m_name][-meta_memory_window:] if model_memories[m_name] else None

            m1, m2, _, t_hps, current_mistakes, hpo_meta = optimize_pipeline(m1_name=m_name,
                                                                             tuning_data=o_train_preprocessed,
                                                                             n_inner_splits=n_inner_splits,
                                                                             n_purged=n_purged,
                                                                             n_embargo=n_embargo,
                                                                             opt_metric=opt_metric,
                                                                             optim_dir='tournament',
                                                                             n_model_trials=n_model_trials,
                                                                             n_trading_trials=n_trading_trials,
                                                                             initial_cash=initial_cash,
                                                                             min_qty=min_qty,
                                                                             max_qty=max_qty,
                                                                             tc_per_unit=tc_per_unit,
                                                                             slippage_per_unit=slippage_per_unit,
                                                                             existing_memory=memory_window,
                                                                             selected_features=selected_features,
                                                                             enable_plotting=False)
            model_memories[m_name].append(current_mistakes)
            model_trading_hps_folds[m_name].append(t_hps)

            # Plot Distribution of Metrics from Optimization
            to_remove = ["variance", "n_trials"]
            bins_dict = {}
            hpo_meta_copy = hpo_meta.copy()
            for k in list(hpo_meta_copy.keys()):
                if k in to_remove:
                    hpo_meta_copy.pop(k)
                else:
                    v = hpo_meta_copy[k]

                    values = np.array(v)
                    cleaned = values[np.isfinite(values)]
                    cleaned_list = cleaned.tolist()
                    hpo_meta_copy[k] = cleaned_list

                    # bins_dict[k] = np.linspace(min(cleaned_list), max(cleaned_list), 31)

                    if not cleaned_list:
                        print(f"Warning: All values for '{k}' are non-finite (inf/-inf/nan).")
                        # hpo_meta_copy.pop(k)
                        bins_dict[k] = np.linspace(0, 1, 31)
                    else:
                        bins_dict[k] = np.linspace(min(cleaned_list), max(cleaned_list), 31)

            plot_multiple_financial_distributions(stat_dict=hpo_meta_copy,
                                                  model_name=m_name,
                                                  bins_dict=bins_dict,
                                                  title='Tournament')

            # 1. Build the Calibration Set from the memory window of past folds, PLUS
            # this fold's own mistakes, ALL RE-SCORED through this fold's just-fitted
            # `m2` model. Previously, inherited memory kept the probabilities that were
            # frozen in by whichever (now-discarded) M2 model generated them originally
            # -- meaning a later fold's calibration set could be scored on a completely
            # different scale than its own live M2 model. Rescoring via `m2` here (and
            # via m2_final inside optimize_pipeline, for the tuple stored back into
            # model_memories) keeps every probability in the calibration set consistent
            # with whatever model is actually judging live signals right now.
            cal_probs = []
            if memory_window:
                for past_feats, past_labs, _past_probs in memory_window:
                    cal_probs.extend(_rescore_mistakes_via_model(past_feats, past_labs, m2))

            if current_mistakes is not None:
                curr_feats, curr_labs, _curr_probs = current_mistakes
                cal_probs.extend(_rescore_mistakes_via_model(curr_feats, curr_labs, m2))

            cal_arr = np.array(cal_probs) if cal_probs else None

            # 2. Evaluate on the Unseen Outer Test Block
            o_test = data.loc[o_test_idx]
            granular_df, benchmark_df, fold_details, conf_preds_tuple = evaluate_on_test_set(m1_model=m1,
                                                                                             m1_name=m_name,
                                                                                             m2_model=m2,
                                                                                             test_data=o_test,
                                                                                             t_hps=t_hps, 
                                                                                             calibration_probs=cal_arr,
                                                                                             selected_features=selected_features,
                                                                                             fitted_preprocessor=fitted_preprocessor,
                                                                                             return_details=True,
                                                                                             num_trials=hpo_meta['n_trials'], # n_model_trials * n_trading_trials
                                                                                             variance_of_sharpes=hpo_meta['variance'],
                                                                                             meta_hpo_psrs=hpo_meta['probabilistic_sharpes'],
                                                                                             initial_cash=initial_cash,
                                                                                             tc_per_unit=tc_per_unit,
                                                                                             slippage_per_unit=slippage_per_unit,
                                                                                             min_qty=min_qty,
                                                                                             max_qty=max_qty)
            
            # Accumulate this model's confidence predicition tuple for this OOS Test Data
            model_conf_preds_folds[m_name].append(conf_preds_tuple)

            granular_leaderboard[m_name].append(granular_df)
            strategy_comparison_leaderboard[m_name].append(benchmark_df)
            full_results[m_name].append(fold_details)

    if model_conf_preds_folds:
        plot_nested_conformal_preds(model_conf_preds_folds, n_outer_splits=n_outer_splits, title_pref='Tournament')
        plot_nested_confusion_matrices(model_conf_preds_folds, n_outer_splits=n_outer_splits, title_pref='Tournament')
        plot_nested_reliability_diagrams(model_conf_preds_folds, n_outer_splits=n_outer_splits, title_pref='Tournament')

    if mi_pi_folds:
        plot_nested_feature_importances(mi_pi_folds=mi_pi_folds, title_pre='Tournament', pi_model=pi_model)
    
    if Xs_folds:
        plot_nested_correlation_heatmap(Xs_folds=Xs_folds, title_pre='Tournament')

    # --- Per Model Backtesting Results over all folds ---
    for m_name in model_names:
        plot_nested_wfv_dashboard(m_name, full_results[m_name], title_pref='tournament')

    # Granular leaderboard should look like this
    # {
    #  RandomForest : [DataFrame of OOS Test Fold 1 with index "pair" and columns "total_return / sharpe / probabilistic_sharpe / ...",
    #                  DataFrame for Fold 2, ...],
    #  GRU : [...]
    # }
    
    # --- Professional Granular Reporting ---
    for m_name in model_names:
        if granular_leaderboard[m_name]:
            print(f"Generating professional LaTeX performance table for {m_name}...")
            generate_performance_table(
                granular_dfs=granular_leaderboard[m_name],
                model_name=m_name,
                phase_name='tournament'
            )
        
        if model_trading_hps_folds[m_name]:
            print(f"Generating professional LaTeX Trading HPs table for {m_name}...")
            generate_trading_hps_table(
                t_hps_list=model_trading_hps_folds[m_name],
                model_name=m_name,
                phase_name='tournament'
            )

    print("Generating professional LaTeX Strategy Comparison table...")
    generate_strategy_comparison_table(
        comparison_data=strategy_comparison_leaderboard,
        phase_name='tournament'
    )

    print("Generating professional LaTeX RRF Ranking tables...")
    generate_rrf_leaderboard(
        consolidated_results=granular_leaderboard,
        phase_name='tournament'
    )

    print("Generating professional LaTeX Tournament Summary tables...")
    generate_tournament_summary_tables(
        consolidated_results=granular_leaderboard,
        phase_name='tournament'
    )


# ==============================================================================
# 4. PHASE 2: PRODUCTION REFINEMENT
# ==============================================================================

def run_wfv(data: pd.DataFrame, 
            global_test_data: pd.DataFrame,
            winner_name: str, 
            n_inner_splits, 
            n_purged, 
            n_embargo, 
            opt_metric, 
            n_model_trials, 
            n_trading_trials,
            mi_thresh=0.005,
            pi_threshold=0.0,
            corr_thresh=0.90,
            pi_model='RF',
            initial_cash=10_000,
            tc_per_unit=0.0001,
            slippage_per_unit=0.0002,
            min_qty=1,
            max_qty=100_000):
    """
    Finalizes the tournament winner by refining hyperparameters and trading logic on the complete dataset.

    METHODOLOGY:
    1. Global Consolidation: Uses the maximum available history to ensure final models see all regimes.
    2. Final Preprocessing & Selection: Re-runs the full FS pipeline on the aggregated dataset to 
        capture the most robust signal set for production.
    3. Final Conformal Optimization: Calls 'optimize_pipeline' to select the absolute best architectural 
        parameters and the optimal Conformal significance level.
    4. Artifact Persistence: Packages M1, M2, the 'selected_features' list, and the final 
        'calibration_set' into a single '.joblib' artifact for production deployment.

    Args:
        data (pd.DataFrame): The raw aggregated forex dataset.
        winner_name (str): The architecture that won the tournament phase.
        n_inner_splits (int): Inner CV folds for final parameter tuning.
        n_purged (int): Overlap removal bars.
        n_embargo (int): Leakage prevention bars.
        opt_metric (str): Goal metric (e.g., 'sharpe').
        n_model_trials (int): Architectural tuning iterations.
        n_trading_trials (int): Risk-management tuning iterations.
        mi_thresh (float): Mutual Information score threshold for feature selection.
        pi_threshold (float): Permutation Importance threshold for feature selection.

    Returns:
        None: Saves the final production artifact to './models/final_production/'.
    """
    os.makedirs("./models/final_production", exist_ok=True)

    # Separate Labels
    labels_full = data[['y_side', 'y_truth', 'pair']].copy()
    y_full = data['y_side']

    # Separate Execution Metadata (OHLC/ATR)
    meta_cols = ['num_atr', 'tx_high', 'tx_low', 'tx_close']
    execution_metadata_full = data[meta_cols].copy().rename(columns={'num_atr': 'raw_atr'})

    # Exclude metadata from ML features to avoid double columns
    X_raw = data.drop(columns=['y_side', 'y_truth', 'pair'] + meta_cols, errors='ignore')

    # Fit preprocessor on full production history
    X_preprocessed, fitted_preprocessor = preprocess_features(X_raw)
    
    # USE COCKPIT VALUES FOR FINAL SELECTION
    X_selected_df, mi_pi_series, Xs_before_after = run_feature_selection_pipeline(X=X_preprocessed,
                                                                                  y=y_full,
                                                                                  model=pi_model, # defaults to RandomForest
                                                                                  corr_thresh=corr_thresh, 
                                                                                  mi_thresh=mi_thresh,
                                                                                  pi_thresh=pi_threshold)
    selected_features = X_selected_df.columns.tolist()

    mi_pi_fold = {'Global Training Fold' : mi_pi_series}
    Xs_fold = {'Global Training Fold' : Xs_before_after}
    plot_nested_feature_importances(mi_pi_folds=mi_pi_fold, title_pre='Production', pi_model=pi_model)
    plot_nested_correlation_heatmap(Xs_folds=Xs_fold, title_pre='Production')

    # Fresh concatenation of labels, preprocessed features, and execution metadata
    data_preprocessed = pd.concat([labels_full, X_preprocessed, execution_metadata_full], axis=1)

    m1, m2, m_hps, t_hps, final_mistakes, hpo_meta = optimize_pipeline(m1_name=winner_name,
                                                                       tuning_data=data_preprocessed, 
                                                                       n_inner_splits=n_inner_splits,
                                                                       n_purged=n_purged,
                                                                       n_embargo=n_embargo,
                                                                       opt_metric=opt_metric,
                                                                       optim_dir='production',
                                                                       n_model_trials=n_model_trials,
                                                                       n_trading_trials=n_trading_trials,
                                                                       initial_cash=initial_cash,
                                                                       min_qty=min_qty,
                                                                       max_qty=max_qty,
                                                                       tc_per_unit=tc_per_unit,
                                                                       slippage_per_unit=slippage_per_unit,
                                                                       selected_features=selected_features,
                                                                       enable_plotting=True)
        
    to_remove = ["variance", "n_trials"]
    bins_dict = {}
    hpo_meta_copy = hpo_meta.copy()
    for k in list(hpo_meta_copy.keys()):
        if k in to_remove:
            hpo_meta_copy.pop(k)
        else:
            v = hpo_meta_copy[k]

            values = np.array(v)
            cleaned = values[np.isfinite(values)]
            cleaned_list = cleaned.tolist()
            hpo_meta_copy[k] = cleaned_list

            # bins_dict[k] = np.linspace(min(cleaned_list), max(cleaned_list), 31)
            
            if not cleaned_list:
                print(f"Warning: All values for '{k}' are non-finite (inf/-inf/nan).")
                # hpo_meta_copy.pop(k)
                bins_dict[k] = np.linspace(0, 1, 31)
            else:
                bins_dict[k] = np.linspace(min(cleaned_list), max(cleaned_list), 31)

    plot_multiple_financial_distributions(stat_dict=hpo_meta_copy,
                                          model_name=winner_name,
                                          bins_dict=bins_dict,
                                          title='Production')
    
    # Extract the final calibration set: the probabilities M2 gave to M1's mistakes
    _, f_labs, f_probs = final_mistakes
    valid_mask = ~np.isnan(f_probs) # Filter out NaNs from the final calibration set
    cal_arr = f_probs[valid_mask & (f_labs == 0)].values # label 0 -> M1 was incorrect (bad trade)

    # 2. Evaluate on the Unseen Global Test Data
    granular_df, benchmark_df, fold_details, conf_preds_tuple = evaluate_on_test_set(m1_model=m1,
                                                                                      m1_name=winner_name,
                                                                                      m2_model=m2,
                                                                                      test_data=global_test_data,
                                                                                      t_hps=t_hps, 
                                                                                      calibration_probs=cal_arr,
                                                                                      selected_features=selected_features,
                                                                                      fitted_preprocessor=fitted_preprocessor,
                                                                                      return_details=True,
                                                                                      num_trials=hpo_meta['n_trials'], # n_model_trials * n_trading_trials
                                                                                      variance_of_sharpes=hpo_meta['variance'],
                                                                                      meta_hpo_psrs=hpo_meta['probabilistic_sharpes'],
                                                                                      initial_cash=initial_cash,
                                                                                      tc_per_unit=tc_per_unit,
                                                                                      slippage_per_unit=slippage_per_unit,
                                                                                      min_qty=min_qty,
                                                                                      max_qty=max_qty)
    
    model_conf_preds_folds = {winner_name : [conf_preds_tuple]}

    if model_conf_preds_folds:
        plot_nested_conformal_preds(model_conf_preds_folds, n_outer_splits=1, title_pref='Production', rows_are_models=False)
        plot_nested_confusion_matrices(model_conf_preds_folds, n_outer_splits=1, title_pref='Production', rows_are_models=False)
        plot_nested_reliability_diagrams(model_conf_preds_folds, n_outer_splits=1, title_pref='Production', rows_are_models=False)

    # --- Final Backtesting Results ---
    plot_nested_wfv_dashboard(winner_name, [fold_details], title_pref='production')

    # --- Professional Granular Reporting ---
    print(f"Generating professional LaTeX performance table for {winner_name} (Production)...")
    generate_performance_table(
        granular_dfs=[granular_df],
        model_name=winner_name,
        phase_name='production'
    )

    print(f"Generating professional LaTeX Trading HPs table for {winner_name} (Production)...")
    generate_trading_hps_table(
        t_hps_list=[t_hps],
        model_name=winner_name,
        phase_name='production'
    )

    print(f"Generating professional LaTeX Strategy Comparison table for {winner_name} (Production)...")
    generate_strategy_comparison_table(
        comparison_data={winner_name: [benchmark_df]},
        phase_name='production'
    )

    print(f"Generating professional LaTeX RRF Ranking tables for {winner_name} (Production)...")
    generate_rrf_leaderboard(
        consolidated_results={winner_name: [granular_df]},
        phase_name='production'
    )

    joblib.dump({
        'm1': m1, 
        'm2': m2, 
        'm_hps': m_hps, 
        't_hps': t_hps, 
        'm_name': winner_name, 
        'selected_features': selected_features,
        'calibration_set': cal_arr,
        'preprocessor': fitted_preprocessor
    }, f"./data/models/final_production/global_{winner_name}_prod.joblib")
    
    print(f"Production Refinement Complete. Global model saved to ./models/final_production/global_{winner_name}_prod.joblib")
