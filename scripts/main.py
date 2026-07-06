# Imports
from datetime import datetime
from utils import aggregate_forex_data, plot_pipeline_structure, load_and_split
from forex_dataloader import get_raw_forex_data
from forex_datacleaner import clean_raw_forex_data
from forex_feature_engineering import engineer_forex_features
from forex_label_generator import generate_forex_labels, count_label_dist
from forex_walk_forward_validation import run_nested_wfv, run_wfv

### COCKPIT ###
n_outer_splits=2        # Number of Block Splits for the Outer Loop of Nested WFV
n_inner_splits=2        # Number of Block Splits for the Inner Loop of Nested WFV,
                        # Doubles down as the Number of CV Splits for Final Model Training
meta_memory_window=3    # The window of past OUTER folds in the nested wfv to consider for the training of the M2 (Meta Model) and
                        # to account for potential Regime changes

n_purged=10             # Number of rows to be removed from the end of each training set
n_embargo=10            # Number of rows to be removed from the beginning of a training set if it is preceeded by a test set
n_model_trials=5        # Number of 'outer' Optuna trials to conduct (computationally expensive because it involves Model training)
n_trading_trials=20     # Number of 'inner' Optuna trials to conduct (conducts n backtests on the same model for fair evaluation)
opt_metric='sharpe'     # Options: 'return', 'sharpe', 'mdd', 'calmar'

mutual_info_threshold = 0.005            # threshold below which features are dropped
permutation_importance_threshold = 0.001 # threshold below which features are dropped
correlation_threshold = 0.90             # absolute correlation threshold for feature dropping
pi_model = 'RF'                          # Model to be used for permutation importance (default is 'RF'; also 'GNB' exists)

### Triple-Barrier Method (Marcos Lopez De Prado) Inputs ###
# The issue is how to select these parameters
# because they directly influence the distribution
# of the labels!
horizon = 5                              # horizon (time step) for labels gen
t_final = 5                              # specifies the vertical line of De Prado's Triple Barrier Labels
atr_lookback = 14                        # number of bars for atr in triple barrier labels
tp_atr_multiplier = 2                    # take profit multiplier
sl_atr_multiplier = 2                    # stop loss multiplier (Risk-reward of 1:1 right now)

### Fixed Trading Parameters (others are optimized by Optuna) ###
min_qty = 1_000                          # minimum position size (probably 1 Micro Lot)
max_qty = 100_000                        # maximum position size (1_000 = Micro Lot, 10_000 = Mini Lot)
tc_per_unit = 0.0001                     # 1 bp x share / contract
slippage_per_unit = 0.0002               # 2 bp x share / contract
initial_cash = 10_000                    # starting cash per pair during backtests
### COCKPIT END ###


def main():
    start_time = datetime.now()

    print("\n" + "="*80)
    print("MASTER FOREX ML PIPELINE - TOURNAMENT & PRODUCTION")
    print(f"STARTED AT: {start_time}")
    print("="*80 + "\n")

    txt = "Which part(s) of the pipeline would you like to run:\n"
    txt += "- Tournament (type t)\n"
    txt += "- Production (type p)\n"
    txt += "- Both (type tp)\n"
    txt += "Input: "

    while True:
        response = input(txt)
        
        if response.lower() in ["t", "p", "tp", "pt"]:
            break
        else:
            print("Oops, that was an invalid response.")
    
    # ------------------------
    # Step 1: Data Acquisition
    # ------------------------
    currencies = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
                  "USDTRY", "USDRUB", "USDINR", "USDKRW", "USDBRL", "USDZAR", "USDMXN", "USDCNH"]
    start_date = "2024-04-28" # Maximum is 2 years of data with my free-tier polygon API key
    get_raw_forex_data(currencies, start_date)

    # -----------------------------
    # Step 2: Cleaning & Validation
    # -----------------------------
    clean_raw_forex_data(incl_viol_out=True)

    # -----------------------------------
    # Step 3: Feature & Label Engineering
    # -----------------------------------
    engineer_forex_features()
    generate_forex_labels(t_final=t_final, 
                          atr_lookback=atr_lookback, 
                          tp_atr_multiplier=tp_atr_multiplier, 
                          sl_atr_multiplier=sl_atr_multiplier)
    count_label_dist()
    
    # -------------------
    # Step 4: Aggregation
    # -------------------
    # Combines all pairwise csv files into one big master csv file
    aggregate_forex_data()
    
    # -------------------------------
    # Step 5: Data-Split (Train/Test)
    # -------------------------------
    # Load the absolute global dataset
    agg_file = "./data/csv_files/forex_master_data/aggregated_complete_data.csv"
    split, global_data = load_and_split(path=agg_file, index_col='date', train_pct=0.9)
    global_train_data = split[0]
    global_test_data = split[1]

    # Visualize Pipeline Structure
    plot_pipeline_structure(global_data, n_outer_splits, n_inner_splits)
    
    # -------------------------------------------------------------------
    # Step 6: Phase 1 - THE TOURNAMENT (Unbiased Architecture Comparison)
    # -------------------------------------------------------------------
    # Answers: "Which model is the most robust across time?"
    print("\n[PHASE 1] Starting Architecture Tournament...")
    models_dict = {"rf" : 'RandomForest', 
                   "et" : 'ExtraTrees', 
                   "xgb" : 'XGBoost', 
                   "lgbm" : 'LightGBM',
                   "lstm" : 'LSTM', 
                   "gru" : 'GRU'}
    tournament_models = list(models_dict.values())

    if response in ["t", "tp", "pt"]:
        run_nested_wfv(data=global_train_data,
                       model_names=tournament_models,
                       n_outer_splits=n_outer_splits,
                       n_inner_splits=n_inner_splits,
                       n_model_trials=n_model_trials,
                       n_trading_trials=n_trading_trials,
                       opt_metric=opt_metric,
                       n_purged=n_purged,
                       n_embargo=n_embargo,
                       meta_memory_window=meta_memory_window,
                       mi_thresh=mutual_info_threshold,
                       pi_threshold=permutation_importance_threshold,
                       corr_thresh=correlation_threshold,
                       pi_model=pi_model,
                       initial_cash=initial_cash,
                       tc_per_unit=tc_per_unit,
                       slippage_per_unit=slippage_per_unit,
                       min_qty=min_qty,
                       max_qty=max_qty)
        
    txt = "Pick a Winner:\n"
    txt += "- RandomForest (type rf)\n"
    txt += "- ExtraTrees (type et)\n"
    txt += "- XGBoost (type xgb)\n"
    txt += "- LightGBM (type lgbm)\n"
    txt += "- LSTM (type lstm)\n"
    txt += "- GRU (type gru)\n"
    txt += "Input: "
    while True:
        winner = input(txt)
    
        if winner.lower() in ["rf", "et", "xgb", "lgbm", "lstm", "gru"]:
            break
        else:
            print("Oops, that was an invalid response.")
    
    print(f"Hooray, we found a Winner: {models_dict[winner.lower()]}")


    # -------------------------------------------------------------
    # Step 7: Phase 2 - PRODUCTION REFINEMENT (Global HP Selection)
    # -------------------------------------------------------------
    # Answers: "What are the absolute best parameters for the winner?"
    winner = 'RandomForest'
    print(f"\n[PHASE 2] Tournament Winner: {winner}")
    print(f"Starting Global Production Refinement for {winner}...")

    if response in ["p", "tp", "pt"]:
        run_wfv(data=global_train_data,
                global_test_data=global_test_data,
                winner_name=winner,
                n_inner_splits=n_inner_splits,
                n_purged=n_purged,
                n_embargo=n_embargo,
                opt_metric=opt_metric,
                n_model_trials=n_model_trials,
                n_trading_trials=n_trading_trials,
                mi_thresh=mutual_info_threshold,
                pi_threshold=permutation_importance_threshold,
                corr_thresh=correlation_threshold,
                pi_model=pi_model,
                initial_cash=initial_cash,
                tc_per_unit=tc_per_unit,
                slippage_per_unit=slippage_per_unit,
                min_qty=min_qty,
                max_qty=max_qty)

    end_time = datetime.now()
    diff = end_time - start_time

    # Extract hours, minutes, seconds
    total_seconds = int(diff.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE")
    print(f"FINISHED AT: {end_time}")
    print(f"TOTAL TIME: {hours}h {minutes}m {seconds}s")
    print("="*80)
    

if __name__ == "__main__":
    main()
