# Imports
from utils import aggregate_forex_data, plot_pipeline_structure
from forex_dataloader import get_raw_forex_data
from forex_datacleaner import clean_raw_forex_data
from forex_feature_engineering import engineer_forex_features
from forex_label_generator import generate_forex_labels
from forex_walk_forward_validation import run_nested_wfv, run_wfv
import pandas as pd
import glob
import os

### COCKPIT ###
n_outer_splits=3        # Number of Block Splits for the Outer Loop of Nested WFV
n_inner_splits=3        # Number of Block Splits for the Inner Loop of Nested WFV,
                        # Doubles down as the Number of CV Splits for Final Model Training
meta_memory_window=3    # The window of past OUTER folds in the nested wfv to consider for the training of the M2 (Meta Model)

n_purged=10             # Number of rows to be removed from the end of each training set
n_embargo=10            # Number of rows to be removed from the beginning of a training set if it is preceeded by a test set
n_model_trials=5        # Number of 'outer' Optuna trials to conduct (computationally expensive because it involves Model training)
n_trading_trials=20     # Number of 'inner' Optuna trials to conduct (conducts n backtests on the same model for fair evaluation)
opt_metric='sharpe'     # Options: 'return', 'sharpe', 'mdd', 'calmar'
### COCKPIT END ###

# Step 5: Data-Split (Train/Test)
# Load the absolute global dataset
agg_file = "./data/csv_files/forex_master_data/aggregated_complete_data.csv"
data = pd.read_csv(agg_file, index_col="date")
data.index = pd.to_datetime(data.index)
data = data.sort_index()

# Visualize Pipeline Structure
plot_pipeline_structure(data, n_outer_splits, n_inner_splits)