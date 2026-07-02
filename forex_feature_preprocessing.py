import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector

# ==============================================================================
# 1. PREPROCESSING ENGINE
# ==============================================================================

def preprocess_features(X: pd.DataFrame, fitted_preprocessor=None) -> tuple:
    """
    Applies preprocessing based on column name prefixes.
    Prefixes supported:
    - num_: StandardScaler
    - ord_: OrdinalEncoder
    - cat_: OneHotEncoder
    - bin_, cyc_, time_, pass_: Passthrough

    Args:
        X (pd.DataFrame): Data to transform.
        fitted_preprocessor (ColumnTransformer, optional): An already fitted preprocessor. 
            If None, a new one is fitted on X.

    Returns:
        tuple: (pd.DataFrame preprocessed_X, ColumnTransformer fitted_preprocessor)
    """
    if fitted_preprocessor is None:
        # Define the transformation logic using naming convention
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), make_column_selector(pattern="^num_*")),
                ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), make_column_selector(pattern="^ord_*")),
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), make_column_selector(pattern="^cat_*")),
                ('time', 'passthrough', make_column_selector(pattern="^time_*")),
                ('cyc', 'passthrough', make_column_selector(pattern="^cyc_*")),
                ('bin', 'passthrough', make_column_selector(pattern="^bin_*")),
                ('pass', 'passthrough', make_column_selector(pattern="^pass_*")),
            ],
            remainder='drop' 
        )
        X_transformed = preprocessor.fit_transform(X)
    else:
        preprocessor = fitted_preprocessor
        X_transformed = preprocessor.transform(X)
    
    # Recover column names
    try:
        new_cols = preprocessor.get_feature_names_out()
        # Clean up the 'num__', 'cat__' prefixes that scikit-learn adds
        new_cols = [c.split('__')[1] if '__' in c else c for c in new_cols]
    except:
        new_cols = X.columns
        
    return pd.DataFrame(X_transformed, index=X.index, columns=new_cols), preprocessor

# ==============================================================================
# 2. FEATURE SELECTION ENGINE (Preserved from original script)
# ==============================================================================

def filter_highly_correlated_features(X: pd.DataFrame,
                                      threshold: float = 0.90) -> pd.DataFrame:
    """Step 1: Filter highly correlated features."""
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    X_reduced = X.drop(columns=to_drop)
    print(f"Correlation Filter: Dropping {len(to_drop)} features with corr > {threshold}")
    
    return X_reduced

def select_by_mutual_info(X: pd.DataFrame,
                          y: pd.Series,
                          threshold: float = 0.005) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Step 2: Select features by Mutual Information Classif score threshold."""
    scores = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    mi_series = pd.Series(scores, index=X.columns).sort_values(ascending=False)
    
    selected_features = mi_series[mi_series > threshold].index.tolist()
        
    print(f"MI Selection: Selected {len(selected_features)} features with score > {threshold}.")
    return X[selected_features], mi_series

def select_by_permutation_importance(X: pd.DataFrame,
                                     y: pd.Series, 
                                     model='RF', 
                                     threshold: float = 0.0, 
                                     n_repeats: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Step 3: Select features by Permutation Importance score."""
    if model == 'RF': # RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42)
    elif model == 'GNB': # Gaussian Naive Bayes
        model = GaussianNB()
    model.fit(X, y)
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1)
    importance_series = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
        
    selected_features = importance_series[importance_series > threshold].index.tolist()
    print(f"Permutation Selection: Selected {len(selected_features)} features.")
    
    return X[selected_features], importance_series

def run_feature_selection_pipeline(X: pd.DataFrame,
                                   y: pd.Series, 
                                   model=None,
                                   corr_thresh=0.90, 
                                   mi_thresh=0.005, 
                                   pi_thresh=0.0) -> Tuple[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Helper to run the full selection pipeline on ALREADY PREPROCESSED data."""
    print("Starting Feature Selection Pipeline...")
    X_reduced = filter_highly_correlated_features(X=X,
                                                  threshold=corr_thresh)
    X_after = X_reduced.copy()
    
    X_reduced, mi_series = select_by_mutual_info(X=X_reduced,
                                                 y=y,
                                                 threshold=mi_thresh)
    
    X_reduced, pi_series = select_by_permutation_importance(X=X_reduced,
                                                            y=y,
                                                            model=model,
                                                            threshold=pi_thresh)
    
    print(f"Pipeline Complete: {X.shape[1]} -> {X_reduced.shape[1]} features.")

    return X_reduced, (mi_series, pi_series), (X, X_after)
