import numpy as np
import pandas as pd
import os
from utils import plot_nested_confusion_matrices

def generate_mock_confusion_data(n_models, n_folds, n_samples=1000):
    """
    Generates synthetic data for testing nested confusion matrices.
    """
    model_names = [f"Model_{i+1}" for i in range(n_models)]
    if n_models >= 3:
        model_names = ["RandomForest", "XGBoost", "GaussianNB"]
        
    model_conf_preds_folds = {m_name: [] for m_name in model_names}
    
    for m_idx, m_name in enumerate(model_names):
        for f_idx in range(n_folds):
            # 1. Base M1 Signals (-1 or 1)
            side_m1 = np.random.choice([-1, 1], size=n_samples)
            
            # 2. Ground Truth (y_truth)
            # Create a scenario where M1 has some skill (e.g., 55% accuracy)
            accuracy_m1 = 0.55
            is_correct = np.random.choice([0, 1], size=n_samples, p=[1-accuracy_m1, accuracy_m1])
            y_truth = side_m1.copy()
            y_truth[is_correct == 0] = -side_m1[is_correct == 0]
            
            # 3. M2 Probabilities (Probability that M1 is correct)
            # High probs for correct signals, low for incorrect
            probs_m2 = np.zeros(n_samples)
            # Correct signals get higher M2 confidence
            probs_m2[is_correct == 1] = np.random.beta(5, 2, size=(is_correct == 1).sum())
            # Incorrect signals get lower M2 confidence
            probs_m2[is_correct == 0] = np.random.beta(2, 5, size=(is_correct == 0).sum())
            
            threshold = 0.6
            alpha = 0.1
            
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
    print("CONFUSION MATRIX VISUALIZATION TEST")
    print("="*60)

    # --- SCENARIO 1: TOURNAMENT (Rows are Models) ---
    n_models_t = 6
    n_folds_t = 2
    print(f"\n[TEST 1] Tournament Case: {n_models_t} Models x {n_folds_t} Folds...")
    tournament_data = generate_mock_confusion_data(n_models_t, n_folds_t)
    
    plot_nested_confusion_matrices(
        model_conf_preds_folds=tournament_data,
        n_outer_splits=n_folds_t,
        title_pref='Tournament_Test',
        rows_are_models=True
    )
    print("   -> Tournament plot saved in ./data/figures/confusion_matrices/")

    # --- SCENARIO 2: PRODUCTION (Single Model/Fold) ---
    n_models_p = 1
    n_folds_p = 1
    print(f"\n[TEST 2] Production Case: {n_models_p} Model x {n_folds_p} Fold...")
    production_data = generate_mock_confusion_data(n_models_p, n_folds_p)
    
    plot_nested_confusion_matrices(
        model_conf_preds_folds=production_data,
        n_outer_splits=n_folds_p,
        title_pref='Production_Test',
        rows_are_models=False
    )
    print("   -> Production plot saved in ./data/figures/confusion_matrices/")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
