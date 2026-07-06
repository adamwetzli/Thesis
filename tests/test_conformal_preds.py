import numpy as np
from utils import plot_nested_conformal_preds

def generate_synthetic_conformal_data(n_models=3, n_folds=5, n_samples_per_fold=200):
    """
    Generate synthetic data for testing the conformal prediction plot.
    
    Parameters:
    - n_models: Number of different models to simulate
    - n_folds: Number of outer CV folds
    - n_samples_per_fold: Number of predictions per model per fold
    
    Returns:
    - model_conf_preds_folds: Dictionary matching the expected structure
    """
    model_conf_preds_folds = {}
    model_names = [f'Model_{chr(65+i)}' for i in range(n_models)]  # Model_A, Model_B, etc.
    
    for model_name in model_names:
        fold_tuples = []
        
        # Different model characteristics (varying performance)
        model_bias = np.random.uniform(-0.3, 0.3)  # Some models are better than others
        
        for fold_idx in range(n_folds):
            # Generate synthetic data
            np.random.seed(hash(f"{model_name}_{fold_idx}") % 2**32)  # Reproducible but different per fold
            
            # Step 1: Generate true market direction (50/50)
            y_truth = np.random.choice([0, 1], size=n_samples_per_fold, p=[0.5, 0.5])
            
            # Step 2: Generate signal with some accuracy (varies by model and fold)
            model_accuracy = 0.6 + model_bias + np.random.uniform(-0.1, 0.1)
            model_accuracy = np.clip(model_accuracy, 0.4, 0.8)  # Keep reasonable
            
            # Signal is correct with probability = model_accuracy
            signal_correct = np.random.rand(n_samples_per_fold) < model_accuracy
            side_m1 = np.where(signal_correct, y_truth, 1 - y_truth)
            
            # Step 3: Generate M2 confidence scores (probability of being correct)
            # Better confidence when the signal is likely correct
            probs_m2 = np.zeros(n_samples_per_fold)
            for i in range(n_samples_per_fold):
                if side_m1[i] == y_truth[i]:  # Correct signal
                    # High confidence typically (but with some variance)
                    probs_m2[i] = np.random.beta(a=5, b=2)
                else:  # Incorrect signal
                    # Lower confidence typically
                    probs_m2[i] = np.random.beta(a=2, b=5)
            
            # Add some noise and correlation
            probs_m2 = np.clip(probs_m2 + np.random.normal(0, 0.05, n_samples_per_fold), 0, 1)
            
            # Step 4: Calculate conformal threshold (varies by fold)
            significance_level = np.random.uniform(0.05, 0.2)
            
            # Calculate conformal threshold using quantile of 1 - probs_m2
            n_calibration = min(100, n_samples_per_fold // 2)
            calibration_scores = 1 - probs_m2[:n_calibration]
            conformal_threshold = np.quantile(calibration_scores, 1 - significance_level)
            conformal_threshold = np.clip(conformal_threshold, 0.1, 0.4)  # Keep reasonable
            
            # Store in required tuple format
            fold_tuples.append((side_m1, y_truth, probs_m2, conformal_threshold, significance_level))
        
        model_conf_preds_folds[model_name] = fold_tuples
    
    return model_conf_preds_folds, model_names

# Example usage with flexible parameters:
if __name__ == "__main__": 
    data_6x2, model_names_6x2 = generate_synthetic_conformal_data(n_models=6, n_folds=2, n_samples_per_fold=200)
    plot_nested_conformal_preds(data_6x2, n_outer_splits=2, title_pref="test_6models_2folds", rows_are_models=True)
    
    data_1x1, model_names_1x1 = generate_synthetic_conformal_data(n_models=1, n_folds=1, n_samples_per_fold=200)
    plot_nested_conformal_preds(data_1x1, n_outer_splits=1, title_pref="test_1model_1fold", rows_are_models=False)
    
    print("\nAll test plots generated successfully!")