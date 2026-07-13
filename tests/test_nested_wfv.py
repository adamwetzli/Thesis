import os
from utils import load_and_split
from forex_walk_forward_validation import run_nested_wfv

# MINI COCKPIT #
# tournament_models = ['RandomForest', 'ExtraTrees', 'XGBoost', 'LightGBM', 'LSTM', 'GRU'] # Subset for testing
tournament_models = ['XGBoost'] # Subset for testing
n_outer_splits = 2      # Number of unbiased "Future" blocks
n_inner_splits = 2      # Number of tuning folds within each train block
n_model_trials = 2      # Very low for rapid testing
n_trading_trials = 20   # Very low for rapid testing

opt_metric = 'sharpe'
n_purged = 10
n_embargo = 10
meta_memory_window = 2

# Feature Selection Thresholds
mi_thresh = 0.005
pi_threshold = 0.001
corr_thresh = 0.90
# MINI COCKPIT END #


def test_tournament_pipeline():
    print("\n" + "="*80)
    print("TESTING ARCHITECTURE TOURNAMENT (run_nested_wfv)")
    print("="*80)

    # 1. Load available master data
    agg_file = "./data/csv_files/forex_master_data/aggregated_complete_data.csv"
    if not os.path.exists(agg_file):
        print(f"Error: {agg_file} not found. Please run the full pipeline or aggregation first.")
        return

    # Load and split (using 90% for tournament as per main)
    split, global_data = load_and_split(path=agg_file, index_col='date', train_pct=0.9)
    global_train_data = split[0]

    print(f"\n[TEST] Starting Tournament for models: {tournament_models}")
    print(f"[TEST] Outer Splits: {n_outer_splits}, Inner Splits: {n_inner_splits}")
    print(f"[TEST] Model Trials: {n_model_trials}, Trading Trials: {n_trading_trials}")
    
    # 3. Execute Tournament
    try:
        leaderboard = run_nested_wfv(data=global_train_data,
                                     model_names=tournament_models,
                                     n_outer_splits=n_outer_splits,
                                     n_inner_splits=n_inner_splits,
                                     n_model_trials=n_model_trials,
                                     n_trading_trials=n_trading_trials,
                                     opt_metric=opt_metric,
                                     n_purged=n_purged,
                                     n_embargo=n_embargo,
                                     meta_memory_window=meta_memory_window,
                                     mi_thresh=mi_thresh,
                                     pi_threshold=pi_threshold,
                                     corr_thresh=corr_thresh)
        
        print("\n" + "="*80)
        print("TOURNAMENT TEST SUCCESSFUL")
        print("="*80)
        
    except Exception as e:
        print(f"\n[TEST FAILED] Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tournament_pipeline()
