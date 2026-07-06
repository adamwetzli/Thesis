from forex_feature_preprocessing import run_feature_selection_pipeline, preprocess_features
from utils import load_and_split

agg_file = "./data/csv_files/forex_master_data/aggregated_complete_data.csv"
split, global_data = load_and_split(path=agg_file, index_col='date', train_pct=0.9)
global_train_data = split[0]

X_raw = global_train_data.drop(columns=['y_side', 'y_truth', 'pair'], errors='ignore')
y_full = global_train_data['y_side']

X_preprocessed = preprocess_features(X_raw)
X_selected_df = run_feature_selection_pipeline(X_preprocessed, y_full, mi_thresh=0.005, enable_plotting=True, plot_title="Production")