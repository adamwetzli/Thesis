import pandas as pd
import numpy as np
import os
from typing import Dict, List
from utils import generate_rrf_leaderboard

# --- Synthetic Data Generation ---
def create_synthetic_data(num_folds=2):
    pairs = ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", 
             "USDCHF", "USDCNH", "USDJPY", "USDMXN", "USDTRY", "USDZAR"]
    
    metrics = ["total_return", "sharpe", "probabilistic_sharpe", 
               "deflated_sharpe", "max_dd", "win_rate", "profit_factor", "cagr"]
    
    folds_data = []
    for _ in range(num_folds):
        # Generate random data for metrics
        data = np.random.rand(len(pairs), len(metrics))
        df = pd.DataFrame(data, index=pairs, columns=metrics)
        # Ensure max_dd is negative as per logic
        df["max_dd"] = -df["max_dd"] 
        folds_data.append(df)
        
    return folds_data

# Create consolidated dictionary
# Mapping models to their respective fold DataFrames
models = ["RandomForest", "ExtraTrees", "XGBoost", "LightGBM", "LSTM", "GRU"]
consolidated_results = {model: create_synthetic_data() for model in models}

# --- Execution ---
# Note: Ensure the 'data/tables/...' directory structure exists or the function will raise an error
# The function will write the LaTeX files to disk
generate_rrf_leaderboard(consolidated_results, phase_name="test", top_n=2)