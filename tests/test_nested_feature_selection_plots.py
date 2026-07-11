from utils import load_and_split, plot_nested_feature_importances, BlockingTimeSeriesSplit
from forex_feature_preprocessing import preprocess_features, run_feature_selection_pipeline
import os

def test_nested_feat_imp(n_outer_splits, title):
    # 1. Load available master data
    agg_file = "./data/csv_files/forex_master_data/aggregated_complete_data.csv"
    if not os.path.exists(agg_file):
        print(f"Error: {agg_file} not found. Please run the full pipeline or aggregation first.")
        return

    # Load and split (using 90% for tournament as per main)
    split, global_data = load_and_split(path=agg_file, index_col='date', train_pct=0.9)
    global_train_data = split[0]

    print(f"[TEST] Outer Splits: {n_outer_splits}")

    mi_pi_folds = {f"Fold {j+1}" : [] for j in range(n_outer_splits)} # Stores the Mutual Info/Permutation Importance scores of all features across all folds
    Xs_folds = {f"Fold {j+1}" : [] for j in range(n_outer_splits)} # Stores the Features before and after correlation filtering across all folds

    outer_cv = BlockingTimeSeriesSplit(n_splits=n_outer_splits)

    for i, (o_train_idx, o_test_idx) in enumerate(outer_cv.split(global_train_data)):
        print(f"\n--- ENTERING OUTER FOLD: {i + 1}/{n_outer_splits} ---")
        
        o_train = global_train_data.loc[o_train_idx]
    
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
                                                                                model="RF")
        mi_pi_folds[f"Fold {i+1}"] = mi_pi_series # Store Scores in correct fold key
        Xs_folds[f"Fold {i+1}"] = Xs_series # Stores the before/after features in correct fold key

    if mi_pi_folds:
        plot_nested_feature_importances(mi_pi_folds=mi_pi_folds, title_pre=title, pi_model="RF")

if __name__ == "__main__":
    # test_nested_feat_imp(2, "test_tournament")
    test_nested_feat_imp(1, "test_production")