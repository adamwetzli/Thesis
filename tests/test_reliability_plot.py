import numpy as np
import pandas as pd
import os
from utils import plot_nested_reliability_diagrams

def generate_mock_data(n_models, n_folds, n_samples=2000):
    """
    Generates synthetic data for complex nested scenarios.
    
    Calibration Profiles:
    - Model 1: Perfectly Calibrated (Institutional baseline)
    - Model 2: Overconfident (Optimistic bias)
    - Model 3: Underconfident (Pessimistic bias)
    - Model 4: Random Noise (No resolution)
    """
    model_names = [f"Arch_{i+1}" for i in range(n_models)]
    if n_models >= 4:
        model_names = ["RandomForest", "XGBoost", "LSTM", "GaussianNB"]
        
    model_conf_preds_folds = {m_name: [] for m_name in model_names}
    
    for m_idx, m_name in enumerate(model_names):
        for f_idx in range(n_folds):
            # 1. Generate M2 Confidence scores (0 to 1)
            # Use beta distribution to create realistic "clusters" of confidence
            probs_m2 = np.random.beta(2, 2, n_samples)
            
            # 2. Assign True Outcome (M1 correctness) based on Profile
            if "RandomForest" in m_name or m_idx == 0:
                # PROFILE: Well-calibrated
                y_true_m2 = np.array([np.random.choice([0, 1], p=[1-p, p]) for p in probs_m2])
            elif "XGBoost" in m_name or m_idx == 1:
                # PROFILE: Overconfident (Actual accuracy < Predicted confidence)
                # Strategy: Accuracy is 70% of whatever is predicted
                acc_probs = np.clip(probs_m2 * 0.7, 0.05, 0.95)
                y_true_m2 = np.array([np.random.choice([0, 1], p=[1-p, p]) for p in acc_probs])
            elif "LSTM" in m_name or m_idx == 2:
                # PROFILE: Underconfident (Actual accuracy > Predicted confidence)
                # Strategy: Accuracy is higher than predicted (e.g., sigmoid shift)
                acc_probs = np.clip(probs_m2 * 1.3, 0.05, 0.95)
                y_true_m2 = np.array([np.random.choice([0, 1], p=[1-p, p]) for p in acc_probs])
            else:
                # PROFILE: Noise (Accuracy is just 50/50 regardless of confidence)
                y_true_m2 = np.random.choice([0, 1], size=n_samples)

            # 3. Create M1 Signal and Ground Truth pairs
            # side_m1 == y_truth if y_true_m2 == 1
            side_m1 = np.random.choice([-1, 1], size=n_samples)
            y_truth = side_m1.copy()
            
            # Flip those where M2 says it should be a mistake
            mistake_mask = (y_true_m2 == 0)
            y_truth[mistake_mask] = -side_m1[mistake_mask]
            
            # 4. Mock Conformal Params
            threshold = 0.8
            alpha = 0.05
            
            conf_preds_tuple = (
                pd.Series(side_m1), 
                pd.Series(y_truth), 
                pd.Series(probs_m2), 
                threshold, 
                alpha
            )
            model_conf_preds_folds[m_name].append(conf_preds_tuple)
            
    return model_conf_preds_folds

if __name__ == "__main__":
    print("="*60)
    print("RELIABILITY DIAGRAM STRESS TEST")
    print("="*60)

    # --- SCENARIO 1: THE TOURNAMENT (Large Grid) ---
    # 4 models, 5 folds = 20 subplots. Tests scaling and label placement.
    n_models_t = 6
    n_folds_t = 3
    print(f"\n[TEST 1] Simulating Tournament: {n_models_t} Models x {n_folds_t} Folds...")
    tournament_data = generate_mock_data(n_models_t, n_folds_t)
    
    plot_nested_reliability_diagrams(
        model_conf_preds_folds=tournament_data,
        n_outer_splits=n_folds_t,
        title_pref='Tournament_StressTest',
        rows_are_models=True
    )
    print("   -> Generated Tournament plot in ./data/figures/conformal_predictions/")

    # --- SCENARIO 2: PRODUCTION RUN (Single Plot) ---
    # 1 model, 1 fold. Tests the 'squeeze=False' robustness and aesthetic sizing.
    n_models_p = 1
    n_folds_p = 1
    print(f"\n[TEST 2] Simulating Production: {n_models_p} Model x {n_folds_p} Fold...")
    production_data = generate_mock_data(n_models_p, n_folds_p)
    
    plot_nested_reliability_diagrams(
        model_conf_preds_folds=production_data,
        n_outer_splits=n_folds_p,
        title_pref='Production_StressTest',
        rows_are_models=False
    )
    print("   -> Generated Production plot in ./data/figures/conformal_predictions/")

    print("\n" + "="*60)
    print("STRESS TEST COMPLETE")
    print("Check ./data/figures/conformal_predictions/ for 'Tournament' vs 'Production' layouts.")
    print("="*60)
