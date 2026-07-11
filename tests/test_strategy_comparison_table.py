import pandas as pd
import numpy as np
import os
from typing import Dict, List
from utils import generate_strategy_comparison_table

def create_synthetic_strategy_data() -> Dict[str, List[pd.DataFrame]]:
    """
    Creates synthetic data mimicking the structure shown in the image.
    Returns a dictionary with model names as keys and lists of fold DataFrames as values.
    """
    # Define the currency pairs
    pairs = ['AUDUSD', 'EURUSD', 'GBPUSD', 'NZDUSD', 'USDCAD', 
             'USDCCHF', 'USDCNH', 'USDJPY', 'USDMXN', 'USDTRY', 'USDZAR']
    
    # Strategy names
    strategies = ['Buy and Hold', 'SMA Crossover', 'M1 Only', 'M1 + M2 (Fixed)', 
                  'M1 + M2 (Global)', 'M1 + M2 (Conformal)']
    
    # Create synthetic data for Fold 1 (based on the image)
    fold1_data = {
        'AUDUSD': [0.56, -1.13, -6.49, -6.54, -0.58, -4.32],
        'EURUSD': [2.70, 0.07, -5.81, -3.49, -5.01, 2.62],
        'GBPUSD': [2.39, -1.28, -9.76, -9.85, -6.43, -5.53],
        'NZDUSD': [0.42, -0.65, -1.82, -1.81, 3.72, 3.39],
        'USDCAD': [0.35, -1.81, 1.19, 3.56, 2.33, 2.46],
        'USDCCHF': [-2.61, -4.05, -7.95, -5.78, 1.37, -0.27],
        'USDCNH': [-1.78, -2.71, 5.95, 1.99, 3.59, 2.80],
        'USDJPY': [-4.08, 0.45, 2.06, 0.86, -1.89, 0.45],
        'USDMXN': [0.09, -1.08, 8.05, 5.55, 1.22, -3.13],
        'USDTRY': [4.55, 3.39, 8.44, 8.48, 6.69, 5.50],
        'USDZAR': [0.53, -0.78, 10.55, 8.96, 8.61, 7.59]
    }
    
    # Create synthetic data for Fold 2 (based on the image)
    fold2_data = {
        'AUDUSD': [3.69, -0.58, -14.77, -9.00, 0.00, 0.00],
        'EURUSD': [1.58, -0.21, -16.49, -11.04, 0.00, 0.00],
        'GBPUSD': [2.18, 0.45, -15.32, -14.47, 0.00, 0.00],
        'NZDUSD': [2.83, 0.25, -21.03, -10.23, 0.00, 0.00],
        'USDCAD': [-2.47, 3.05, -17.84, -17.82, 0.00, -1.35],
        'USDCCHF': [-0.68, -0.09, -19.80, -6.18, 0.00, 0.00],
        'USDCNH': [-6.02, 3.58, 0.43, 0.43, 0.00, 0.00],
        'USDJPY': [1.59, -2.06, -0.43, -0.43, 0.00, 0.00],
        'USDMXN': [-4.52, 3.04, 16.82, 16.82, 2.17, 1.28],
        'USDTRY': [4.48, 3.50, 10.97, 10.97, 5.55, 14.65],
        'USDZAR': [-3.52, 3.27, 5.66, 5.66, 3.50, 4.00]
    }
    
    # Convert to DataFrames
    fold1_df = pd.DataFrame(fold1_data, index=strategies).T
    fold2_df = pd.DataFrame(fold2_data, index=strategies).T
    
    # Create synthetic data for additional models (variations of the original)
    def add_noise_to_fold(fold_df: pd.DataFrame, noise_scale: float = 0.5) -> pd.DataFrame:
        """Add random noise to create variations for different models"""
        noise = np.random.normal(0, noise_scale, fold_df.shape)
        return fold_df + noise
    
    # Create data for multiple models
    comparison_data = {}
    
    # Model 1: Original data (exact match from image)
    comparison_data['Model_1'] = [fold1_df.copy(), fold2_df.copy()]
    
    # Model 2: Slight variations
    comparison_data['Model_2'] = [
        add_noise_to_fold(fold1_df, 0.3),
        add_noise_to_fold(fold2_df, 0.3)
    ]
    
    # Model 3: More significant variations
    comparison_data['Model_3'] = [
        add_noise_to_fold(fold1_df, 0.8),
        add_noise_to_fold(fold2_df, 0.8)
    ]
    
    # Model 4: Different pattern (more variation)
    comparison_data['Model_4'] = [
        add_noise_to_fold(fold1_df, 1.2) + np.random.normal(0, 0.5, fold1_df.shape),
        add_noise_to_fold(fold2_df, 1.2) + np.random.normal(0, 0.5, fold2_df.shape)
    ]
    
    return comparison_data


# Example usage
if __name__ == "__main__":
    # Create synthetic data
    print("Creating synthetic data...")
    comparison_data = create_synthetic_strategy_data()
    
    # Generate the tables
    print("Generating strategy comparison tables...")
    generate_strategy_comparison_table(comparison_data, phase_name='test')
    
    print("\nDone! Tables have been generated in the 'data/tables/test/strategy_comparison/' directory.")