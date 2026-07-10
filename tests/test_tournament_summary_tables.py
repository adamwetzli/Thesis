import pandas as pd
import numpy as np
import os
from typing import Dict, List
from utils import generate_tournament_summary_tables

def generate_synthetic_data(models: List[str], n_folds: int = 2) -> Dict[str, List[pd.DataFrame]]:
    """
    Generates synthetic data for trading model metrics across multiple folds.
    """
    metrics = [
        'total_return', 'sharpe', 'probabilistic_sharpe', 'max_dd', 
        'cagr', 'win_rate', 'profit_factor', 'n_trades', 
        'avg_capital_exposure', 'avg_trade_size', 'deflated_sharpe', 
        'pfdr', 'm2_brier'
    ]
    
    consolidated_results = {}
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    for model in models:
        model_folds = []
        for _ in range(n_folds):
            # Generate random data for each metric
            # Using different distributions to simulate realistic values

            data = {
                'total_return': np.random.uniform(0.05, 0.50),
                'sharpe': np.random.normal(1.5, 0.5),
                'probabilistic_sharpe': np.random.uniform(0.70, 0.99),
                'max_dd': np.random.uniform(-0.2, -0.02),
                'cagr': np.random.uniform(0.05, 0.40),
                'win_rate': np.random.uniform(0.45, 0.65),
                'profit_factor': np.random.normal(1.8, 0.3),
                'n_trades': np.random.randint(50, 200),
                'avg_capital_exposure': np.random.uniform(10, 50),
                'avg_trade_size': np.random.uniform(5, 20),
                'deflated_sharpe': np.random.uniform(0.3, 0.8),
                'm2_brier': np.random.uniform(0.1, 0.3)
            }
            # Create a dataframe with 1 row per fold as implied by your function
            model_folds.append(pd.DataFrame([data]))
            
        consolidated_results[model] = model_folds
        
    return consolidated_results

# --- Usage Example ---
if __name__ == "__main__":
    # Define your model names
    my_models = ['LSTM', 'XGBoost', 'RandomForest', "ExtraTrees", "LightGBM", "GRU"]
    
    # Generate the data
    data = generate_synthetic_data(my_models, n_folds=2)
    
    # Verify the structure
    print(f"Generated data for {len(data)} models.")
    
    # You can now pass 'data' directly to your function:
    generate_tournament_summary_tables(data, phase_name='test')