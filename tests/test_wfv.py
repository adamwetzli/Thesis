import pandas as pd
import numpy as np
import os
from utils import load_and_split
from forex_walk_forward_validation import run_wfv

# MINI COCKPIT #
winner_model = 'RandomForest' # Fast for testing
n_inner_splits = 2
n_purged = 10
n_embargo = 10
opt_metric = 'sharpe'

# Low trials for speed
n_model_trials = 1 
n_trading_trials = 100

# Thresholds from Cockpit
mi_thresh = 0.005
pi_threshold = 0.001
corr_thresh = 0.90

# END MINI COCKPIT #

def test_production_pipeline():
    print("\n" + "="*80)
    print("TESTING PRODUCTION REFINEMENT PIPELINE (run_wfv)")
    print("="*80)

    # 1. Load available master data
    agg_file = "./data/csv_files/forex_master_data/aggregated_complete_data.csv"
    if not os.path.exists(agg_file):
        print(f"Error: {agg_file} not found. Please run the full pipeline or aggregation first.")
        return

    # Load and split (using 90% for training/tuning as per main)
    split, global_data = load_and_split(path=agg_file, index_col='date', train_pct=0.9)
    global_train_data = split[0]
    global_test_data = split[1]

    print(f"\n[TEST] Starting Production Refinement for {winner_model}...")
    print(f"[TEST] Model Trials: {n_model_trials}, Trading Trials: {n_trading_trials}")
    
    # 3. Execute Production Refinement
    # This calls optimize_pipeline with enable_plotting=True internally
    try:
        run_wfv(data=global_train_data,
                global_test_data=global_test_data,
                winner_name=winner_model,
                n_inner_splits=n_inner_splits,
                n_purged=n_purged,
                n_embargo=n_embargo,
                opt_metric=opt_metric,
                n_model_trials=n_model_trials,
                n_trading_trials=n_trading_trials,
                mi_thresh=mi_thresh,
                pi_threshold=pi_threshold,
                corr_thresh=corr_thresh)
        
        print("\n" + "="*80)
        print("TEST SUCCESSFUL")
        print(f"Artifact saved to: ./data/models/final_production/global_{winner_model}_prod.joblib")
        print("Check figures/optimization/ for Optuna plots.")
        print("="*80)
        
    except Exception as e:
        print(f"\n[TEST FAILED] Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_production_pipeline()
