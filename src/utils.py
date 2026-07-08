"""
Institutional Utility Module
============================

This module provides core utilities for data management, institutional-grade
cross-validation, and high-fidelity visualization of trading pipeline results.

Core Components:
1. Data Aggregation: Consolidates multi-instrument datasets for global processing.
2. BlockingTimeSeriesSplit: Implementation of Marcos Lopez de Prado's cross-validation 
   methodology (Purging & Embargoing).
3. Pipeline Visualization: Graphical representation of temporal split structures.
4. Performance Analysis: Visualization of backtest results, feature selection impact, 
   and hyperparameter optimization.
"""

import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Union, Any

# ==============================================================================
# 1. DATA MANAGEMENT UTILITIES
# ==============================================================================

def aggregate_forex_data() -> None:
    """
    Consolidates individual currency pair CSVs into a single master dataset.
    
    Loads all files from './data/csv_files/forex_complete_data/', adds an instrument
    identifier column ('pair'), and merges them into a unified 'aggregated_complete_data.csv'.
    Uses memory-safe insertion to prevent DataFrame fragmentation warnings.
    """
    input_dir = "./data/csv_files/forex_complete_data"
    output_dir = "./data/csv_files/forex_master_data"
    output_file = f"{output_dir}/aggregated_complete_data.csv"
    
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(f"{input_dir}/*.csv")
    files = [f for f in files if "aggregated_complete_data.csv" not in f]

    if not files:
        print("!!! No complete data files found to aggregate.")
        return

    print("\n" + "="*80)
    print("MASTER DATA AGGREGATION")
    print(f"Source: {input_dir}")
    print("="*80)
    
    all_data = []
    for file in files:
        pair = os.path.basename(file).replace("_complete_data.csv", "")
        # Load and assign the pair identifier
        df = pd.read_csv(file)
        df['pair'] = pair
        # Explicit copy() consolidates memory blocks and silences fragmentation warnings
        all_data.append(df.copy())
        print(f"   ... Aggregated {pair} (Rows: {len(df)})")
    
    aggregated_df = pd.concat(all_data, ignore_index=True)
    
    # Standardize index and sorting
    aggregated_df['date'] = pd.to_datetime(aggregated_df['date'])
    aggregated_df = aggregated_df.sort_values(by=['date', 'pair'])
    
    aggregated_df.to_csv(output_file, index=False)
    print("\n" + "="*80)
    print("AGGREGATION COMPLETE")
    print(f"Total Master Rows: {len(aggregated_df)}")
    print(f"Saved to:          {output_file}")
    print("="*80)

def load_and_split(path: str, index_col: str, train_pct: float) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """
    Loads a dataset and performs an OOS (Out-of-Sample) global split.
    
    Args:
        path: Path to the CSV file.
        index_col: Column name to use as index.
        train_pct: Fraction of data (0.0 - 1.0) to assign to the training set.

    Returns:
        Tuple: ((global_train_data, global_test_data), full_dataframe)
    """
    print(f"\n[STEP 5] Loading global dataset from {path}...")
    data = pd.read_csv(path, index_col=index_col)
    
    try:
        data.index = pd.to_datetime(data.index)
    except Exception as e:
        print(f"!!! Warning: Could not convert index to datetime: {e}")
    
    data = data.sort_index()

    # Split based on unique timestamps to handle multi-pair datasets correctly
    unique_indices = np.array(data.index.unique().tolist())
    split_idx = int(len(unique_indices) * train_pct)
    split_date = unique_indices[split_idx]

    global_train_data = data.loc[:split_date]
    global_test_data = data.loc[split_date:]

    print(f"   ... Data split complete:")
    print(f"   ... Training Set: {len(global_train_data)} rows (up to {split_date})")
    print(f"   ... Testing Set:  {len(global_test_data)} rows (from {split_date})")

    return (global_train_data, global_test_data), data

# ==============================================================================
# 2. INSTITUTIONAL CROSS-VALIDATION
# ==============================================================================

class BlockingTimeSeriesSplit:
    """
    Blocking Time Series Split with Purging and Embargo.
    
    Implements Marcos Lopez de Prado's best practices for financial time series CV.
    Unlike expanding windows, this creates non-overlapping blocks to minimize 
    leakage and applies temporal buffers.
    
    Components:
    1. Purging: Removes training observations at the end of a block that could
       leak information into the following test set (overlap of labels).
    2. Embargoing: Removes training observations at the beginning of a block
       following a test set (correlation leak).
    """
    
    def __init__(self, n_splits: int, n_purged: int = 0, n_embargo: int = 0):
        """
        Args:
            n_splits: Number of blocks to split the data into.
            n_purged: Number of bars removed from the end of training sets.
            n_embargo: Number of bars removed from the start of training sets (if preceded by test).
        """
        self.n_splits = n_splits
        self.n_purged = n_purged
        self.n_embargo = n_embargo
    
    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits
    
    def split(self, X: pd.DataFrame, y: Any = None, groups: Any = None):
        """
        Generates indices for training and testing blocks.
        
        Args:
            X: Input DataFrame (must have a unique index per bar).
        """
        unique_indices = np.array(X.index.unique().tolist())
        n_samples = len(unique_indices)
        if n_samples < self.n_splits:
            raise ValueError(f"Insufficient samples ({n_samples}) for {self.n_splits} splits.")
            
        k_fold_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            # Define block boundaries
            start = i * k_fold_size
            stop = (i + 1) * k_fold_size if i < self.n_splits - 1 else n_samples
            
            # Sub-split within block: 80% Train / 20% Test
            mid = int(0.8 * (stop - start)) + start
            
            # Apply Purging (End of Training)
            train_end = mid - self.n_purged
            
            # Apply Embargoing (Start of Training - if follows a previous fold)
            train_start = start
            if i > 0:
                train_start += self.n_embargo
            
            # Test set boundaries
            test_start = mid
            test_end = stop
            
            if train_start >= train_end:
                # Fallback for degenerate cases (massive purging on small data)
                yield unique_indices[train_start:train_start], unique_indices[test_start:test_end]
            else:
                yield unique_indices[train_start:train_end], unique_indices[test_start:test_end]

# ==============================================================================
# 3. PIPELINE VISUALIZATION
# ==============================================================================

def plot_pipeline_structure(data: pd.DataFrame, 
                            n_outer_splits: int, 
                            n_inner_splits: int, 
                            split_frac: float = 0.9) -> None:
    """
    Generates a high-fidelity visualization of the Nested Walk-Forward structure.
    
    Args:
        data: Global DataFrame.
        n_outer_splits: Number of tournament folds.
        n_inner_splits: Number of tuning folds.
        split_frac: Global train/test split ratio.
    """
    h1 = n_outer_splits * (n_inner_splits + 4.5)
    h2 = n_inner_splits + 1.5
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 16), sharex=True, 
                                   gridspec_kw={'height_ratios': [h1, h2], 'hspace': 0.15})
    
    all_dates = data.index.unique().sort_values()
    start_date, end_date = all_dates[0], all_dates[-1]
    time_span = end_date - start_date
    ax_padding = time_span * 0.2
    
    split_idx = int(len(all_dates) * split_frac)
    cutoff_date = all_dates[split_idx]
    train_dates = all_dates[:split_idx]
    test_dates = all_dates[split_idx:]
    
    fold_h = 0.8
    legend_elements = []

    # --- Phase 1: The Tournament ---
    ax1.set_title("Phase 1: Nested Walk-Forward Tournament (Architecture Comparison)", 
                 fontsize=24, fontweight='bold', pad=25)
    
    outer_cv = BlockingTimeSeriesSplit(n_splits=n_outer_splits)
    
    for o_idx, (o_train_indices, o_test_indices) in enumerate(outer_cv.split(pd.DataFrame(index=train_dates))):
        y_base = o_idx * (n_inner_splits + 4.5)
        
        # Shared Preprocessing Indicator
        box_y = y_base + n_inner_splits + 2.5
        box_x = o_train_indices[len(o_train_indices)//2]
        bbox_props = dict(boxstyle="round,pad=0.5", fc="aliceblue", ec="darkblue", lw=1.5, alpha=0.9)
        fs_text = "1. Preprocessing (Scaling)\n2. Feature Selection (Once per Outer Fold)\n3. Nested HPO"
        
        ax1.text(box_x, box_y, fs_text, ha='center', va='center', ma='left',
                 bbox=bbox_props, fontsize=19, fontweight='bold', color='darkblue')
        
        bracket_y = box_y - 1.2
        ax1.plot([o_train_indices[0], o_train_indices[-1]], [bracket_y, bracket_y], color='darkblue', lw=2)
        ax1.plot([o_train_indices[0], o_train_indices[0]], [bracket_y, bracket_y - 0.5], color='darkblue', lw=2)
        ax1.plot([o_train_indices[-1], o_train_indices[-1]], [bracket_y, bracket_y - 0.5], color='darkblue', lw=2)
        ax1.annotate("", xy=(box_x, box_y - 0.75), xytext=(box_x, bracket_y),
                     arrowprops=dict(arrowstyle="->", color="darkblue", lw=1.5))

        # Visualizing Folds
        art_ot = ax1.fill_between(o_train_indices, y_base, y_base + fold_h, color='blue', alpha=0.15, hatch='//')
        if o_idx == 0: legend_elements.append((art_ot, 'Outer Training Set'))
        
        art_test = ax1.fill_between(o_test_indices, y_base, y_base + n_inner_splits + fold_h, color='red', alpha=0.3)
        if o_idx == 0: legend_elements.append((art_test, 'Outer Test Set (OOS)'))
        
        inner_cv = BlockingTimeSeriesSplit(n_splits=n_inner_splits)
        for i_idx, (i_train_indices, i_val_indices) in enumerate(inner_cv.split(pd.DataFrame(index=o_train_indices))):
            y_pos = y_base + 1 + i_idx
            art_tr = ax1.fill_between(i_train_indices, y_pos, y_pos + fold_h, color='blue', alpha=0.5)
            art_val = ax1.fill_between(i_val_indices, y_pos, y_pos + fold_h, color='orange', alpha=0.7)
            if o_idx == 0 and i_idx == 0:
                legend_elements.append((art_tr, 'Inner Training Set'))
                legend_elements.append((art_val, 'Inner Validation Set'))
            ax1.text(start_date, y_pos + fold_h/2, f"INNER FOLD {i_idx+1} ", va='center', ha='right', fontsize=19, color='grey')
        ax1.text(start_date, y_base + fold_h/2, f"OUTER FOLD {o_idx+1} ", va='center', ha='right', fontweight='bold', fontsize=19, color='darkblue')

    max_y_phase1 = (n_outer_splits - 1) * (n_inner_splits + 4.5) + n_inner_splits + fold_h
    ax1.axvline(cutoff_date, color='black', linestyle='--', linewidth=2)
    ax1.fill_between(test_dates, 0, max_y_phase1, color='grey', alpha=0.1)
    ax1.set_yticks([])
    ax1.set_ylim(-1, max_y_phase1 + 3)
    ax1.set_xlim(start_date - ax_padding, end_date)

    # --- Phase 2: Production Refinement ---
    ax2.set_title("Phase 2: Global Production Refinement (Winner Validation)", fontsize=24, fontweight='bold', pad=15)
    ax2.text(all_dates[len(all_dates)//2], n_inner_splits + 0.3, 
             "Consolidated Global Preprocessing & Winner Calibration", 
             ha='center', fontsize=19, color='darkred', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="darkred", lw=1))

    prod_cv = BlockingTimeSeriesSplit(n_splits=n_inner_splits)
    for p_idx, (p_train_indices, p_val_indices) in enumerate(prod_cv.split(pd.DataFrame(index=train_dates))):
        y_pos = p_idx
        ax2.fill_between(p_train_indices, y_pos, y_pos + fold_h, color='blue', alpha=0.5)
        ax2.fill_between(p_val_indices, y_pos, y_pos + fold_h, color='orange', alpha=0.7)
        ax2.text(start_date, y_pos + fold_h/2, f"INNER FOLD {p_idx+1} ", va='center', ha='right', fontsize=19, color='grey')
    
    ax2.fill_between(test_dates, 0, (n_inner_splits - 1) + fold_h, color='red', alpha=0.3)
    ax2.axvline(cutoff_date, color='black', linestyle='--', linewidth=2)
    ax2.set_yticks([])
    ax2.set_xlabel("Time", fontsize=14, labelpad=10)
    ax2.set_ylim(-0.5, n_inner_splits + 1.2)
    
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=12, maxticks=24))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=14)

    handles = [e[0] for e in legend_elements]
    labels = [e[1] for e in legend_elements]
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=True, edgecolor='black', bbox_to_anchor=(0.5, 0.035), fontsize=24)

    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    os.makedirs("data/figures/methodology", exist_ok=True)
    plt.savefig("./data/figures/methodology/pipeline_structure.png", bbox_inches='tight', dpi=300)
    plt.show()

# ==============================================================================
# 4. ANALYTICAL VISUALIZATION
# ==============================================================================

def plot_nested_correlation_heatmap(Xs_folds: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], title_pre: str):
    """
    Plots a correlation heatmap of before and after features for each outer fold.
    """
    n_folds = len(Xs_folds)
    if n_folds == 0:
        return
    
    # Find max features for sizing
    max_features = 0
    for X_before, X_after in Xs_folds.values():
        max_features = max_features if max_features > len(X_before.columns) else len(X_before.columns)
        max_features = max_features if max_features > len(X_after.columns) else len(X_after.columns)
    
    # Size cells appropriately
    cell_size = 0.15
    heatmap_size = min(max_features * cell_size, 16)
    
    width = heatmap_size * 2 + 2.2
    height_per_heatmap = heatmap_size
    
    fig, axes = plt.subplots(n_folds, 2, 
                             figsize=(width, height_per_heatmap * n_folds), 
                             squeeze=False)
    
    # Apply subplot adjustments FIRST so positions are calculated correctly
    plt.subplots_adjust(left=0.20, right=0.92, top=0.91, bottom=0.08, wspace=0.74, hspace=0.48)
    
    for i, (fold_name, (X_before, X_after)) in enumerate(Xs_folds.items()):
        corr_before = X_before.corr().abs()
        corr_after = X_after.corr().abs()
        
        # Get the actual position of the left subplot for this row
        ax_position = axes[i, 0].get_position()
        
        # Calculate vertical center of this specific row
        row_center = (ax_position.y0 + ax_position.y1) / 2
        
        # Place the fold label exactly at the center of this row
        fig.text(0.01, row_center, fold_name, 
                rotation=90, ha='center', va='center', 
                fontsize=30, fontweight='bold', color='darkblue',
                transform=fig.transFigure)
        
        # Left Column: Correlation heatmap before
        ax_before = axes[i, 0]
        heatmap_before = sns.heatmap(corr_before, ax=ax_before, cmap='YlOrRd', 
                                    square=True, 
                                    cbar_kws={'shrink': 0.65, 'aspect': 20, 'label': '', 'pad': 0.05})
        
        # Colorbar label font
        cbar_before = heatmap_before.collections[0].colorbar
        cbar_before.set_label('Absolute Correlation', fontsize=26, fontweight='bold')
        cbar_before.ax.tick_params(labelsize=22)
        
        ax_before.set_title(f"BEFORE\n({len(X_before.columns)} Features)", fontsize=25)
        
        # Show FEWER feature names but make them LARGER
        n_features_before = len(X_before.columns)
        if n_features_before > 60:
            step = max(1, n_features_before // 12)
            label_fontsize = 24
        elif n_features_before > 40:
            step = max(1, n_features_before // 10)
            label_fontsize = 26
        elif n_features_before > 20:
            step = max(1, n_features_before // 8)
            label_fontsize = 28
        else:
            step = max(1, n_features_before // 6)
            label_fontsize = 30
        
        ax_before.set_xticks(range(0, n_features_before, step))
        ax_before.set_yticks(range(0, n_features_before, step))
        ax_before.set_xticklabels(X_before.columns[::step], rotation=90, fontsize=label_fontsize)
        ax_before.set_yticklabels(X_before.columns[::step], fontsize=label_fontsize)
        
        # Right Column: Correlation heatmap after
        ax_after = axes[i, 1]
        heatmap_after = sns.heatmap(corr_after, ax=ax_after, cmap='YlOrRd', 
                                   square=True, 
                                   cbar_kws={'shrink': 0.65, 'aspect': 20, 'label': '', 'pad': 0.05})
        
        cbar_after = heatmap_after.collections[0].colorbar
        cbar_after.set_label('Absolute Correlation', fontsize=26, fontweight='bold')
        cbar_after.ax.tick_params(labelsize=22)
        
        ax_after.set_title(f"AFTER\n({len(X_after.columns)} Features)", fontsize=25)
        
        n_features_after = len(X_after.columns)
        if n_features_after > 60:
            step = max(1, n_features_after // 12)
            label_fontsize = 22
        elif n_features_after > 40:
            step = max(1, n_features_after // 10)
            label_fontsize = 24
        elif n_features_after > 20:
            step = max(1, n_features_after // 8)
            label_fontsize = 26
        else:
            step = max(1, n_features_after // 6)
            label_fontsize = 28
        
        ax_after.set_xticks(range(0, n_features_after, step))
        ax_after.set_yticks(range(0, n_features_after, step))
        ax_after.set_xticklabels(X_after.columns[::step], rotation=90, fontsize=label_fontsize)
        ax_after.set_yticklabels(X_after.columns[::step], fontsize=label_fontsize)
    
    # Title handling
    fold_txt = 'Global Training Fold' if n_folds == 1 else 'Folds'
    
    # Wrap long title
    import textwrap
    full_title = f"Correlations in {title_pre} Before vs. After Filtering across {fold_txt}"
    wrapped_title = "\n".join(textwrap.wrap(full_title, width=80))

    # Suptitle perfectly centered across entire figure (x=0.5 is figure center)
    fig.suptitle(wrapped_title, fontsize=35, fontweight='bold', y=0.94, x=0.5, ha='center')
    
    os.makedirs("/data/figures/feature_selection", exist_ok=True)
    save_path = f"./data/figures/feature_selection/{title_pre}_nested_correlation_heatmap.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   ... Nested correlation heatmap plot saved to {save_path}")


def plot_nested_feature_importances(mi_pi_folds: Dict[str, Tuple[pd.Series, pd.Series]], title_pre, pi_model, 
                                    top_k: int = 15) -> None:
    """
    Plots Mutual Information and Permutation Importance scores for each outer fold.
    
    Args:
        mi_pi_folds: Dictionary mapping fold names to (mi_series, pi_series).
        top_k: Number of top features to display in each plot.
    """
    n_folds = len(mi_pi_folds)
    if n_folds == 0:
        return

    # Width 22, balanced right margin, generous left margin for vertical labels
    fig, axes = plt.subplots(n_folds, 2, figsize=(22, 6 * n_folds), squeeze=False)
    
    # Calculate the vertical position for each fold label
    # This properly centers each label vertically within its corresponding row
    for i, (fold_name, (mi_series, pi_series)) in enumerate(mi_pi_folds.items()):
        
        # Account for the actual subplot area (which doesn't include the title area)
        ax_position = axes[i, 0].get_position()
        row_center_figure = (ax_position.y0 + ax_position.y1) / 2
        
        fig.text(0.02, row_center_figure, fold_name, 
                rotation=90, ha='center', va='center', 
                fontsize=20, fontweight='bold', color='darkblue',
                transform=fig.transFigure)
        
        # Left Column: Mutual Information
        ax_mi = axes[i, 0]
        top_mi = mi_series.sort_values(ascending=False).head(top_k)
        sns.barplot(x=top_mi.values, y=top_mi.index, ax=ax_mi, palette='viridis', hue=top_mi.index, legend=False)
        ax_mi.set_title("")
        ax_mi.set_xlabel("Mutual Info Score", fontsize=15, fontweight='bold')
        ax_mi.set_ylabel("")  
        
        # Right Column: Permutation Importance
        ax_pi = axes[i, 1]
        top_pi = pi_series.sort_values(ascending=False).head(top_k)
        sns.barplot(x=top_pi.values, y=top_pi.index, ax=ax_pi, palette='magma', hue=top_pi.index, legend=False)
        ax_pi.set_title("")
        ax_pi.set_xlabel("Permutation Importance Score", fontsize=15, fontweight='bold')
        ax_pi.set_ylabel("") 

    # For depicted model, that was used in permutation importance, in the figure title
    if pi_model == 'RF':
        model_txt = 'RandomForest'
    elif pi_model == 'GNB':
        model_txt = 'Gaussian Naive Bayes'

    # To adjust for the nested or production case
    if len(mi_pi_folds) == 1:
        fold_txt = 'Global Training Fold'
    else:
        fold_txt = 'Folds'

    # Balanced rect to center the subplots grid as much as possible
    plt.subplots_adjust(left=0.14, right=0.95, top=0.85, bottom=0.08, wspace=0.4, hspace=0.4)
    fig.suptitle(f"[{title_pre}] Mutual Information and Permutation Importance\n[{model_txt}] across {fold_txt}", 
             fontsize=20, fontweight='bold', y=0.98, x=0.5, ha='center')
    
    os.makedirs("data/figures/feature_selection", exist_ok=True)
    save_path = f"./data/figures/feature_selection/{title_pre}_nested_feature_importances_{model_txt}.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')  # Added bbox_inches='tight'
    plt.close()
    print(f"   ... Nested feature importance plot saved to {save_path}")

def plot_optuna_study(study: Any, m1_name: str, title_suffix: str = "", optim_cat='tournament') -> None:
    """Generates and saves Optuna study analytics (History, Importance, Coordinates)."""
    try:
        import optuna.visualization as vis
        import plotly.io as pio
        
        os.makedirs(f"data/figures/optimization/{optim_cat}/{m1_name}", exist_ok=True)
        s_clean = title_suffix.lower().replace(" ", "_")
        
        fig_hist = vis.plot_optimization_history(study)
        # Save as HTML (interactive)
        pio.write_html(fig_hist, f"data/figures/optimization/{optim_cat}/{m1_name}/opt_history_{s_clean}.html")
        # Save as high-res PNG
        pio.write_image(fig_hist, f"data/figures/optimization/{optim_cat}/{m1_name}/opt_history_{s_clean}.png", 
                        width=1200, height=800, scale=3)
        try:
            fig_imp = vis.plot_param_importances(study)
            pio.write_html(fig_imp, f"data/figures/optimization/{optim_cat}/{m1_name}/param_importance_{s_clean}.html")
            pio.write_image(fig_imp, f"data/figures/optimization/{optim_cat}/{m1_name}/param_importance_{s_clean}.png", 
                            width=1200, height=800, scale=3)
        except Exception: pass

        fig_par = vis.plot_parallel_coordinate(study)
        pio.write_html(fig_par, f"data/figures/optimization/{optim_cat}/{m1_name}/parallel_coordinate_{s_clean}.html")
        pio.write_image(fig_par, f"data/figures/optimization/{optim_cat}/{m1_name}/parallel_coordinate_{s_clean}.png", 
                       width=1200, height=800, scale=3)
        
        print(f"   ... Optuna analytics saved to data/figures/optimization/{optim_cat}/{m1_name}/ (HTML + PNG)")

    except ImportError:
        print("!!! Warning: plotly/kaleido missing. Optuna plots skipped.")

def plot_backtest_results(ohlc: pd.DataFrame, 
                          trade_history: List[Dict], 
                          equity_history: List[Dict], 
                          initial_cash: float, 
                          stats: Optional[Dict] = None, 
                          filename: str = "Backtest Results.png") -> None:
    """
    Institutional Backtest Report Plot.
    Visualizes Equity curve, trade PnL bubbles, and order execution markers.
    """
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)
    x_min, x_max = ohlc.index.min(), ohlc.index.max()
    padding = (x_max - x_min) * 0.02
    for axis in ax: axis.set_xlim([x_min - padding, x_max + padding])

    # 1. Orders
    ax[0].set_title("Execution Log")
    ax[0].plot(ohlc.index, ohlc["close"], color="blue", alpha=0.4, label='Price')
    for trade in trade_history:
        color = "green" if trade["direction"] == "long" else "red"
        e_marker = "^" if trade["direction"] == "long" else "v"
        x_marker = "v" if trade["direction"] == "long" else "^"
        ax[0].scatter(trade["entry_time"], trade["entry_price_raw"], marker=e_marker, color=color, s=30)
        ax[0].scatter(trade["exit_time"], trade["exit_price_raw"], marker=x_marker, color=color, s=30, alpha=0.6)

    # 2. Trade PnL Bubbles
    if trade_history:
        t_df = pd.DataFrame(trade_history)
        t_df["exit_time"] = pd.to_datetime(t_df["exit_time"])
        grouped = t_df.groupby("exit_time").agg({"net_pnl": "sum", "size": lambda x: (x.abs() * t_df.loc[x.index, "entry_price_slippage"]).sum()})
        pnl_pct = (grouped["net_pnl"] / (grouped["size"] + 1e-7)) * 100
        ax[1].set_title("Trade Return Profile (%)")
        ax[1].scatter(grouped.index, pnl_pct, s=np.abs(grouped["net_pnl"])*2+10, 
                      c=["green" if x >= 0 else "red" for x in pnl_pct], alpha=0.6, edgecolors="black")
        ax[1].axhline(0, color='black', lw=1, alpha=0.5)
        ax[1].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=2))

    # 3. Cumulative Equity
    eq_df = pd.DataFrame(equity_history).set_index('timestamp')
    strat_ret = eq_df['equity'] / initial_cash
    bench_ret = ohlc["close"] / ohlc["close"].iloc[0]
    
    ax[2].plot(strat_ret.index, strat_ret, color="purple", lw=1.5, label='Strategy (MTM)')
    ax[2].plot(ohlc.index, bench_ret, color="black", lw=1, alpha=0.5, label='Benchmark')
    ax[2].fill_between(strat_ret.index, 1.0, strat_ret, where=(strat_ret >= 1.0), color="green", alpha=0.1)
    ax[2].fill_between(strat_ret.index, 1.0, strat_ret, where=(strat_ret < 1.0), color="red", alpha=0.1)
    ax[2].set_title("Cumulative Performance (Relative to Initial)")
    ax[2].legend(loc='upper left')

    if stats:
        s_text = (f"Return: {stats.get('total_return', 0):>8.2%}\nSR: {stats.get('sharpe', 0):>8.2f}\n"
                  f"PSR: {stats.get('probabilistic_sharpe', 0):>8.2%}\nDSR: {stats.get('deflated_sharpe', 0):>8.2%}\n"
                  f"MDD: {stats.get('max_dd', 0):>8.2%}\n"
                  f"Trades: {stats.get('n_trades', 0):>8}\nWin%: {stats.get('win_rate', 0):>8.2%}")
        ax[2].text(0.02, 0.05, s_text, transform=ax[2].transAxes, fontsize=8.5, bbox=dict(facecolor='white', alpha=0.8), family='monospace')

    plt.tight_layout(rect=[0, 0.08, 1, 0.94])       
    os.makedirs("data/figures", exist_ok=True)
    plt.savefig(f"./data/figures/{filename}")
    plt.show()

def plot_nested_wfv_dashboard(model_name: str, 
                              all_fold_results: List[Dict[str, Dict]], 
                              title_pref: str = 'tournament',
                              max_pairs: int = 3) -> None:
    """
    Generates high-resolution tournament dashboards for a specific architecture.
    Splits into multiple figures if number of pairs exceeds max_pairs.
    """
    if not all_fold_results: return
    
    n_folds = len(all_fold_results)
    all_pairs = list(all_fold_results[0].keys())
    n_total_pairs = len(all_pairs)
    
    num_parts = (n_total_pairs + max_pairs - 1) // max_pairs
    
    for part_idx in range(num_parts):
        start_idx = part_idx * max_pairs
        end_idx = min((part_idx + 1) * max_pairs, n_total_pairs)
        pairs = all_pairs[start_idx:end_idx]
        n_pairs = len(pairs)
        
        rows_per_pair = 3
        total_rows = n_pairs * rows_per_pair
        
        # ========== DYNAMIC FIGURE SIZING ==========
        # Base sizes
        base_width_per_fold = 6
        base_height_per_row = 3.5
        
        # Scale down for many pairs/folds
        width_scale = min(1.0, 20 / (n_folds * base_width_per_fold))
        height_scale = min(1.0, 30 / (total_rows * base_height_per_row))
        
        fig_width = max(12, n_folds * base_width_per_fold * width_scale)
        fig_height = max(8, total_rows * base_height_per_row * height_scale)
        
        # ========== DYNAMIC FONT SIZES ==========
        # Base font sizes that scale with number of elements
        if n_pairs <= 3:
            title_font = 26
            fold_font = 18
            label_font = 15
            pair_font = 14
            stats_font = 13
            legend_font = 16
            tick_font = 9
            legend_ncol = 4
            top_margin = 0.92
            legend_y = 0.96
            title_y = 0.98
        elif n_pairs <= 5:
            title_font = 22
            fold_font = 16
            label_font = 8
            pair_font = 12
            stats_font = 7
            legend_font = 10
            tick_font = 8
            legend_ncol = 5
            top_margin = 0.925
            legend_y = 0.955
            title_y = 0.97
        elif n_pairs <= 8:
            title_font = 18
            fold_font = 14
            label_font = 10
            pair_font = 11
            stats_font = 6.5
            legend_font = 9
            tick_font = 7
            legend_ncol = 4
            top_margin = 0.92
            legend_y = 0.95
            title_y = 0.965
        elif n_pairs <= 12:
            title_font = 16
            fold_font = 10
            label_font = 5
            pair_font = 10
            stats_font = 5.5
            legend_font = 8
            tick_font = 6
            legend_ncol = 4
            top_margin = 0.93
            legend_y = 0.95
            title_y = 0.96
        else:  # >12 pairs
            title_font = 14
            fold_font = 9
            label_font = 4.5
            pair_font = 9
            stats_font = 4
            legend_font = 7
            tick_font = 6
            legend_ncol = 3
            top_margin = 0.9175
            legend_y = 0.9375
            title_y = 0.95
        
        # Create figure
        fig, axes = plt.subplots(nrows=total_rows, ncols=n_folds, 
                                 figsize=(fig_width, fig_height),
                                 squeeze=False, sharex='col')
        
        # Global Title
        part_text = f" (Part {part_idx + 1}/{num_parts})" if num_parts > 1 else ""
        fig.suptitle(f"{title_pref.title()} Performance: {model_name.upper()}{part_text}", 
                     fontsize=title_font, fontweight='bold', y=title_y)

        # Global Legend
        legend_elements = [
            Line2D([0], [0], color='blue', alpha=0.4, lw=1.5, label='Price'),
            Line2D([0], [0], color='purple', lw=2.0, label='Strategy (MTM)'),
            Line2D([0], [0], color='black', lw=1.0, alpha=0.4, label='Benchmark'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='green', label='Enter Long', markersize=8, linestyle='None'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='green', alpha=0.6, label='Exit Long', markersize=8, linestyle='None'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='red', label='Enter Short', markersize=8, linestyle='None'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='red', alpha=0.6, label='Exit Short', markersize=8, linestyle='None'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', alpha=0.6, label='Profit Bubble', markersize=8, linestyle='None', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', alpha=0.6, label='Loss Bubble', markersize=8, linestyle='None', markeredgecolor='black'),
        ]
        
        fig.legend(handles=legend_elements, loc='upper center', ncol=legend_ncol, 
                   fontsize=legend_font, frameon=True, bbox_to_anchor=(0.5, legend_y),
                   edgecolor='grey', fancybox=True, shadow=False)

        for f_idx, fold_data in enumerate(all_fold_results):
            for p_idx, pair in enumerate(pairs):
                res = fold_data.get(pair)
                if not res: continue
                
                ohlc = res['ohlc']
                trade_history = res['trade_history']
                equity_df = res['equity_history']
                stats = res['stats']
                initial_cash = res['initial_cash']
                
                ax_exec = axes[p_idx * rows_per_pair + 0, f_idx]
                ax_pnl  = axes[p_idx * rows_per_pair + 1, f_idx]
                ax_cum  = axes[p_idx * rows_per_pair + 2, f_idx]
                
                # --- 1. Execution Log ---
                ax_exec.plot(ohlc.index, ohlc["close"], color="blue", alpha=0.3, linewidth=0.8 if n_pairs > 8 else 1.5)
                
                # Adjust marker sizes for many pairs
                marker_size = max(10, min(25, 30 - n_pairs))
                for t in trade_history:
                    color = "green" if t["direction"] == "long" else "red"
                    e_marker = "^" if t["direction"] == "long" else "v"
                    x_marker = "v" if t["direction"] == "long" else "^"
                    ax_exec.scatter(t["entry_time"], t["entry_price_raw"], marker=e_marker, color=color, s=marker_size)
                    ax_exec.scatter(t["exit_time"], t["exit_price_raw"], marker=x_marker, color=color, s=marker_size, alpha=0.6)
                ax_exec.grid(True, alpha=0.3)

                # --- 2. Trade PnL Bubbles ---
                if trade_history:
                    t_df = pd.DataFrame(trade_history)
                    t_df["exit_time"] = pd.to_datetime(t_df["exit_time"])
                    grouped = t_df.groupby("exit_time").agg({
                        "net_pnl": "sum", 
                        "size": lambda x: (x.abs() * t_df.loc[x.index, "entry_price_slippage"]).sum()
                    })
                    pnl_pct = (grouped["net_pnl"] / (grouped["size"] + 1e-7)) * 100
                    # Scale bubble sizes for many pairs
                    bubble_scale = max(5, min(15, 20 - n_pairs // 2))
                    ax_pnl.scatter(grouped.index, pnl_pct, s=np.abs(grouped["net_pnl"])*0.5 + bubble_scale, 
                                   c=["green" if x >= 0 else "red" for x in pnl_pct], 
                                   alpha=0.6, edgecolors="black")
                ax_pnl.axhline(0, color='black', lw=0.8, alpha=0.5)
                ax_pnl.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
                ax_pnl.grid(True, alpha=0.3)

                # --- 3. Cumulative Performance ---
                if not equity_df.empty:
                    strat_ret = equity_df['equity'] / initial_cash
                    bench_ret = ohlc["close"] / ohlc["close"].iloc[0]
                    
                    ax_cum.plot(strat_ret.index, strat_ret, color="purple", lw=1.0 if n_pairs > 8 else 1.5)
                    ax_cum.plot(ohlc.index, bench_ret, color="black", lw=0.8, alpha=0.4)
                    ax_cum.fill_between(strat_ret.index, 1.0, strat_ret, where=(strat_ret >= 1.0), color="green", alpha=0.1)
                    ax_cum.fill_between(strat_ret.index, 1.0, strat_ret, where=(strat_ret < 1.0), color="red", alpha=0.1)
                    
                    if stats:
                        if n_pairs <= 5:
                            s_text = (f"Return: {stats.get('total_return', 0):.2%}\n"
                                      f"Sharpe: {stats.get('sharpe', 0):.2f}\n"
                                      f"PSR: {stats.get('probabilistic_sharpe', 0):.2%}\n"
                                      f"DSR: {stats.get('deflated_sharpe', 0):.2%}\n"
                                      f"MDD: {stats.get('max_dd', 0):.2%}\n"
                                      f"CAGR: {stats.get('cagr', 0):.2%}\n"
                                      f"Win%: {stats.get('win_rate', 0):.2%}\n"
                                      f"PFactor: {stats.get('profit_factor', 0):.2f}\n"
                                      f"Trades: {stats.get('n_trades', 0)}\n"
                                      f"Avg Exp: {stats.get('avg_capital_exposure', 0):.1f}%\n"
                                      f"Avg Size: {stats.get('avg_trade_size', 0):.1f}%")
                        else:
                            s_text = (f"DSR: {stats.get('deflated_sharpe', 0):.2%}\n"
                                      f"Sharpe: {stats.get('sharpe', 0):.2f}\n"
                                      f"MDD: {stats.get('max_dd', 0):.2%}\n"
                                      f"Avg Exp: {stats.get('avg_capital_exposure', 0):.1f}%\n"
                                      f"Avg Size: {stats.get('avg_trade_size', 0):.1f}%")
                        
                        stats_x = 0.02
                        stats_y = 0.05
                        
                        ax_cum.text(stats_x, stats_y, s_text, transform=ax_cum.transAxes, fontsize=stats_font, 
                                    family='monospace', verticalalignment='bottom',
                                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='grey', pad=1))

                ax_cum.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5 if n_pairs > 8 else 8))
                ax_cum.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax_cum.get_xticklabels(), rotation=90, ha='center', fontsize=tick_font)
                
                for a in [ax_exec, ax_pnl, ax_cum]:
                    a.tick_params(axis='both', which='major', labelsize=tick_font)
                    a.grid(True, alpha=0.2)

        # Apply tight_layout
        plt.tight_layout(rect=[0.08, 0.02, 0.98, top_margin])
        
        # Add side labels (left side) AFTER layout
        for f_idx in range(n_folds):
            if f_idx == 0:
                for p_idx, pair in enumerate(pairs):
                    ax_exec = axes[p_idx * rows_per_pair + 0, f_idx]
                    ax_pnl = axes[p_idx * rows_per_pair + 1, f_idx]
                    ax_cum = axes[p_idx * rows_per_pair + 2, f_idx]
                    
                    bbox_exec = ax_exec.get_position()
                    bbox_pnl = ax_pnl.get_position()
                    bbox_cum = ax_cum.get_position()
                    
                    # Adjust label X positions based on number of pairs
                    label_x_offset = -0.065 if n_pairs <= 5 else (-0.033 if n_pairs <= 8 else -0.035)
                    pair_x_offset = -0.1 if n_pairs <= 5 else (-0.0555 if n_pairs <= 8 else -0.045)
                    
                    fig.text(bbox_exec.x0 + label_x_offset, bbox_exec.y0 + bbox_exec.height/2, 
                            "Execution\nLog", rotation=90, ha='center', va='center', 
                            fontsize=label_font, fontweight='bold')
                    
                    fig.text(bbox_pnl.x0 + pair_x_offset, bbox_pnl.y0 + bbox_pnl.height/2, 
                            pair, rotation=90, ha='center', va='center', 
                            fontsize=pair_font, fontweight='bold', color='darkred')

                    fig.text(bbox_pnl.x0 + label_x_offset, bbox_pnl.y0 + bbox_pnl.height/2, 
                            'Trade Return\nProfile', rotation=90, ha='center', va='center', 
                            fontsize=label_font, fontweight='bold')
                    
                    fig.text(bbox_cum.x0 + label_x_offset, bbox_cum.y0 + bbox_cum.height/2, 
                            'Cumulative\nPerformance', rotation=90, ha='center', va='center', 
                            fontsize=label_font, fontweight='bold')
        
        # Add column titles (top)
        for f_idx in range(n_folds):
            if n_folds == 1:
                txt = 'Global Test Fold'
            else:
                txt = f"Fold {f_idx+1}"
            
            ax_first = axes[0, f_idx]
            bbox = ax_first.get_position()
            
            # Dynamic title Y offset
            title_y_offset = 0.01 if n_pairs <= 5 else (-0.015 if n_pairs <= 8 else -0.02)
            title_y = bbox.y0 + bbox.height + title_y_offset
            
            fig.text(bbox.x0 + bbox.width/2, title_y, txt, 
                     ha='center', va='bottom', fontsize=fold_font, 
                     fontweight='bold', color='darkblue')
        
        os.makedirs("data/figures/backtests", exist_ok=True)
        suffix = f"_part{part_idx + 1}" if num_parts > 1 else ""
        save_path = f"./data/figures/backtests/{title_pref}_results_{model_name}{suffix}.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"   ... Tournament dashboard saved to {save_path}")


def plot_multiple_financial_distributions(stat_dict, model_name, bins_dict=None, figsize=None, show_stats=True,
                                          title=''):
    """
    Create a symmetric multi-plot figure with multiple financial statistic distributions.
    """
    
    n_stats = len(stat_dict)
    
    if n_stats == 0:
        print("No statistics provided")
        return None
    
    # Determine grid dimensions (as close to square as possible)
    n_cols = int(np.ceil(np.sqrt(n_stats)))
    n_rows = int(np.ceil(n_stats / n_cols))
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easy indexing (handle single plot case)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Store handles for figure legend (will collect from first subplot)
    legend_handles = []
    legend_labels = []
    legend_created = False

    # Plot each distribution
    for idx, (stat_name, values_list) in enumerate(stat_dict.items()):
        ax = axes[idx]
        
        # Process values
        values = np.array(values_list).flatten()
        values = values[~np.isnan(values)]
        
        if len(values) == 0:
            ax.text(0.5, 0.5, f"No valid {stat_name} values", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(stat_name)
            continue
        
        # Determine bins
        if bins_dict and stat_name in bins_dict:
            bins = bins_dict[stat_name]
        else:
            # Automatic bin selection using Freedman-Diaconis rule
            iqr = np.percentile(values, 75) - np.percentile(values, 25)
            if iqr == 0:
                bins = 20
            else:
                bin_width = 2 * iqr / (len(values) ** (1/3))
                bins = max(10, min(50, int((values.max() - values.min()) / bin_width)))
        
        # Plot histogram (prevent auto-legend)
        counts, bin_edges, patches = ax.hist(values, bins=bins, 
                                             edgecolor='black', 
                                             color='steelblue', 
                                             alpha=0.7,
                                             density=False,
                                             label='_nolegend_')
        
        # Add KDE
        has_density = False
        try:
            kde = scipy_stats.gaussian_kde(values)
            x_range = np.linspace(values.min(), values.max(), 200)
            kde_values = kde(x_range) * len(values) * (bin_edges[1] - bin_edges[0])
            density_line = ax.plot(x_range, kde_values, 'r-', linewidth=2, label='_nolegend_')[0]
            has_density = True
        except:
            pass
        
        # Add statistical annotations
        if show_stats and len(values) > 0:
            mean_val = np.mean(values)
            median_val = np.median(values)
            p25 = np.percentile(values, 25)
            p75 = np.percentile(values, 75)
            
            # Add vertical lines
            mean_line = ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
            median_line = ax.axvline(median_val, color='green', linestyle='--', linewidth=1.2, alpha=0.7)
            p25_line = ax.axvline(p25, color='blue', linestyle='--', linewidth=1.2, alpha=0.7)
            p75_line = ax.axvline(p75, color='purple', linestyle='--', linewidth=1.2, alpha=0.7)
            
            # Collect handles and labels from the first subplot only
            if not legend_created:
                from matplotlib.patches import Patch
                hist_patch = Patch(facecolor='steelblue', 
                                  edgecolor='black', 
                                  alpha=0.7,
                                  label='Distribution')
                
                legend_handles = [hist_patch, mean_line, median_line, p25_line, p75_line]
                legend_labels = ['Distribution', 'Mean', 'Median', '25th %ile', '75th %ile']
                
                if has_density:
                    legend_handles.insert(1, density_line)
                    legend_labels.insert(1, 'Density Estimate')
                
                legend_created = True
        
        # Customize subplot
        ax.set_xlim(min(values), max(values))
        ax.set_xlabel(stat_name, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(stat_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
    # Turn off any unused subplots
    for idx in range(n_stats, len(axes)):
        axes[idx].set_visible(False)
    
    # Add legend INSIDE the figure at the bottom
    if legend_handles:
        fig.legend(handles=legend_handles, 
                  labels=legend_labels,
                  loc='lower center',
                  bbox_to_anchor=(0.5, 0.02),  
                  ncol=min(5, len(legend_handles)),
                  frameon=True,
                  fancybox=True,
                  shadow=True,
                  fontsize=10)
    
    # Add title - slightly increased spacing from subplots
    fig.suptitle(f'Distribution of Financial Statistics Across Optimization Trials [{model_name}]', 
                 fontsize=16, fontweight='bold', y=0.96)  
    
    plt.subplots_adjust(bottom=0.1, top=0.92)  
    os.makedirs("data/figures/optimization/distributions", exist_ok=True)
    save_path = f"./data/figures/optimization/distributions/{title}_{model_name}.png"
    plt.savefig(save_path)
    plt.close()


def plot_nested_conformal_preds(model_conf_preds_folds, n_outer_splits, title_pref, threshold=0.5, rows_are_models=True):
    """
    Creates separate figures per fold showing conformal prediction boundaries for all models.
    Each figure shows 3 panels per model/fold combination.
    """
    
    model_names = list(model_conf_preds_folds.keys())
    n_models = len(model_names)
    n_folds = n_outer_splits
    
    # Loop over folds to create individual figures
    for j in range(n_folds):
        if rows_are_models:
            fig_n_rows = n_models
            fig_n_cols = 3
        else:
            fig_n_rows = 1
            fig_n_cols = n_models * 3
        
        # Wider figure to accommodate 3 panels
        fig, axes = plt.subplots(fig_n_rows, fig_n_cols, figsize=(5.5 * fig_n_cols, 5.5 * fig_n_rows), squeeze=False)
        
        for i, model_name in enumerate(model_names):
            fold_tuples = model_conf_preds_folds.get(model_name, [])
            if j >= len(fold_tuples): continue
            
            (side_m1, y_truth, probs_m2, conformal_threshold, significance) = fold_tuples[j]
            
            # Subplot Selection (A, B, C) based on layout
            if rows_are_models:
                ax_a, ax_b, ax_c = axes[i, 0], axes[i, 1], axes[i, 2]
            else:
                ax_a, ax_b, ax_c = axes[0, i * 3], axes[0, i * 3 + 1], axes[0, i * 3 + 2]
            
            # Data Cleaning & Masking
            if hasattr(side_m1, 'values'): side_m1 = side_m1.values
            if hasattr(y_truth, 'values'): y_truth = y_truth.values
            if hasattr(probs_m2, 'values'): probs_m2 = probs_m2.values
            
            valid_mask = ~(np.isnan(probs_m2) | np.isnan(side_m1) | np.isnan(y_truth))
            active_mask = (side_m1 != 0)
            plot_mask = valid_mask & active_mask
            
            if plot_mask.sum() == 0:
                for ax in [ax_a, ax_b, ax_c]:
                    ax.text(0.5, 0.5, 'No Signal Data', ha='center', va='center', fontsize=10)
                continue
            
            side_m1_c = side_m1[plot_mask]
            y_truth_c = y_truth[plot_mask]
            probs_m2_c = probs_m2[plot_mask]
            
            # Directional Probability P(Long)
            probs_dir = np.where(side_m1_c == 1, probs_m2_c, 1 - probs_m2_c)
            y_sig_binary = (side_m1_c == 1).astype(int)
            y_truth_binary = (y_truth_c == 1).astype(int)
            
            # --- Sigmoid Fit & Midpoint Calculation ---
            m2_range = np.linspace(0, 1, 300)
            x_mid = None
            if len(np.unique(y_truth_binary)) > 1:
                try:
                    log_reg = LogisticRegression(C=1e5, solver='lbfgs', max_iter=100)
                    log_reg.fit(probs_dir.reshape(-1, 1), y_truth_binary)
                    
                    # Midpoint: x where logit(p) = 0 => beta0 + beta1*x = 0 => x = -beta0/beta1
                    coef = log_reg.coef_[0][0]
                    intercept = log_reg.intercept_[0]
                    if abs(coef) > 1e-4:
                        x_mid = -intercept / coef
                    
                    y_probs = log_reg.predict_proba(m2_range.reshape(-1, 1))[:, 1]
                    for ax in [ax_a, ax_b]:
                        ax.plot(m2_range, y_probs, color='black', linewidth=1.5, zorder=5, alpha=0.8)
                        if x_mid is not None and 0 <= x_mid <= 1:
                            ax.axvline(x_mid, color='purple', linestyle='--', alpha=0.5, linewidth=1)
                except: pass
            
            # --- Common Plotting Elements ---
            for ax in [ax_a, ax_b]:
                ax.axvline(conformal_threshold, color='darkorange', linestyle=':', alpha=0.7, linewidth=1.5)
                ax.axvline(1 - conformal_threshold, color='darkorange', linestyle=':', alpha=0.7, linewidth=1.5)
                ax.axhline(0.5, color='forestgreen', linestyle='--', alpha=0.3, linewidth=1) # 50% line
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.1, 1.1)
                ax.grid(True, alpha=0.1, linestyle='--')
            
            # --- PANEL A: SIGNAL vs CONFIDENCE (Correctness Colored) ---
            is_correct = (side_m1_c == y_truth_c)
            is_accepted = (probs_m2_c >= conformal_threshold)
            
            # Colors: Hit=Green, Miss=Dark Red, Rej=Grey
            colors_a = np.where(~is_accepted, '#95a5a6', # Rejected (Grey)
                       np.where(is_correct, '#2ecc71',   # Hit (Green)
                                            '#8b0000')) # Miss (Dark Red)
            
            ax_a.scatter(probs_dir, y_sig_binary, c=colors_a, alpha=0.5, s=30, zorder=3, edgecolors='none')
            
            # Row labels
            if rows_are_models:
                ax_a.set_ylabel(f'{model_name}\nSignal Direction', fontsize=21, fontweight='bold')
            else:
                ax_a.set_ylabel(f'GLOBAL TEST FOLD {j+1}\nSignal Direction', fontsize=13, fontweight='bold')
            
            # --- PANEL B: GROUND TRUTH vs CONFIDENCE ---
            mask_l = (y_truth_binary == 1)
            mask_s = (y_truth_binary == 0)
            ax_b.scatter(probs_dir[mask_l], np.ones(mask_l.sum()), color='royalblue', alpha=0.4, s=30, zorder=3)
            ax_b.scatter(probs_dir[mask_s], np.zeros(mask_s.sum()), color='crimson', alpha=0.4, s=30, zorder=3)
            if rows_are_models:
                ax_b.set_ylabel('Ground Truth', fontsize=21, fontweight='bold')
            else:
                ax_b.set_ylabel('Ground Truth', fontsize=13, fontweight='bold')
            
            # --- PANEL C: ERROR MAP (Signal - Truth) ---
            error = side_m1_c - y_truth_c
            ax_c.scatter(probs_dir, error, c=colors_a, alpha=0.5, s=35, zorder=3)
            ax_c.axhline(0, color='black', linewidth=1, alpha=0.4)
            ax_c.set_ylim(-2.5, 2.5)
            if rows_are_models:
                ax_c.set_ylabel('Signal - Truth Error', fontsize=21, fontweight='bold')
            else:
                ax_c.set_ylabel('Signal - Truth Error', fontsize=13, fontweight='bold')
            ax_c.set_yticks([-2, 0, 2])
            ax_c.set_yticklabels(['F-Short', 'Correct', 'F-Long'])
            ax_c.grid(True, alpha=0.1, axis='y')
            
            # --- TITLES & FORMATTING ---
            # Group titles (Column headers)
            if rows_are_models:
                if i == 0:
                    ax_b.set_title(f'FOLD {j+1}', fontsize=18, fontweight='bold', pad=35)
                    ax_a.set_title("Panel A: Signal Accuracy", fontsize=18, color='black', pad=20)
                    ax_b.text(0.5, 1.05, "Panel B: Ground Truth", transform=ax_b.transAxes, ha='center', fontsize=18, color='black')
                    ax_c.set_title("Panel C: Error Map", fontsize=18, color='black', pad=10)
            else:
                # Titles for every model in the row
                ax_b.set_title(f'{model_name}', fontsize=16, fontweight='bold', pad=35)
                ax_a.set_title("Panel A: Signal Accuracy", fontsize=11, color='black', pad=10)
                ax_b.text(0.5, 1.05, "Panel B: Ground Truth", transform=ax_b.transAxes, ha='center', fontsize=11, color='black')
                ax_c.set_title("Panel C: Error Map", fontsize=11, color='black', pad=10)
            
            # Bottom labels
            if rows_are_models:
                if i == n_models - 1:
                    for ax in [ax_a, ax_b, ax_c]: ax.set_xlabel('Directional Meta-Confidence (M2)', fontsize=18)
            else:
                # Every model in the row gets x-labels
                for ax in [ax_a, ax_b, ax_c]: ax.set_xlabel('Directional Meta-Confidence (M2)', fontsize=15)
            
            # Stats for Panel A
            hits = (is_accepted & is_correct).sum()
            miss = (is_accepted & ~is_correct).sum()
            rej = (~is_accepted).sum()
            mid_txt = f'Mid:{x_mid:.2f}' if x_mid is not None else 'Mid:N/A'
            stats_txt = f'Hits:{hits} Miss:{miss}\nRej:{rej}\nα:{significance:.3f}\n{mid_txt}'
            
            ax_a.text(0.98, 0.05, stats_txt, 
                     transform=ax_a.transAxes, fontsize=18, ha='right', va='bottom',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='lightgrey'))

        # Global Legend for each figure
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', label='Correct Signal (Accepted)', markersize=9),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#8b0000', label='Incorrect Signal (Accepted)', markersize=9),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#95a5a6', label='Rejected by Conformal', markersize=9),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', label='Truth: Long', markersize=9),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson', label='Truth: Short', markersize=9),
            Line2D([0], [0], color='black', lw=1.5, label='Sigmoid Fit (Truth Calibration)'),
            Line2D([0], [0], color='forestgreen', linestyle='--', label='50% Probability Threshold'),
            Line2D([0], [0], color='purple', linestyle='--', label='Midpoint (Intersection)'),
            Line2D([0], [0], color='darkorange', linestyle=':', label='Conformal Rejection Zone Boundary')
        ]
        if rows_are_models:
            legend_font = 18
        else:
            legend_font = 12
        fig.legend(handles=legend_elements, loc='upper center', ncol=3, frameon=True, 
                   framealpha=1.0, edgecolor='black', bbox_to_anchor=(0.5, 0.025), fontsize=legend_font)

        plt.suptitle(f'Conformal Signal Analysis: {title_pref} - Fold {j+1}', 
                     fontsize=22, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        os.makedirs("data/figures/conformal_predictions", exist_ok=True)
        if n_folds > 1:
            save_path = f"./data/figures/conformal_predictions/{title_pref}_conformal_predictions_part{j+1}.png"
        else:
            save_path = f"./data/figures/conformal_predictions/{title_pref}_conformal_predictions.png"
        
        plt.savefig(save_path, dpi=250, bbox_inches='tight')
        plt.close()
        print(f"   ... Enhanced Conformal analysis plot (Fold {j+1}) saved to {save_path}")


def plot_nested_confusion_matrices(model_conf_preds_folds, n_outer_splits, title_pref, rows_are_models=True):
    """
    Creates confusion matrices for M1, M2, and Final predictions across all folds.
    
    Layout: 3 Panels per Fold
    - Panel A: M1 Base Model (Actual vs Predicted) - 2x2 binary
    - Panel B: M2 Meta-Model (M1 Correctness vs M2 Prediction) - 2x2 binary
    - Panel C: Final Filtered (Actual vs Predicted) - 2x2 binary (Accepted signals only)
    """
    from sklearn.metrics import confusion_matrix
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    model_names = list(model_conf_preds_folds.keys())
    n_models = len(model_names)
    n_folds = n_outer_splits
    
    if rows_are_models:
        y_label_font = 20
        ax_title_font = 20
        x_label_font = 20
        top_margin = 0.02
        stats_font = 18
        bottom_margin = 0.055
        annot_fontsize = 18
        tick_font = 12
    else:
        y_label_font = 16
        ax_title_font = 16
        x_label_font = 12
        bottom_margin = 0.03
        top_margin = 0.05
        stats_font = 10
        annot_fontsize = 18
        tick_font = 14

    def add_colorbar(im, ax):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

    for j in range(n_folds):
        if rows_are_models:
            fig_n_rows = n_models
            fig_n_cols = 3
        else:
            fig_n_rows = 1
            fig_n_cols = n_models * 3
        
        # Create figure with subplots
        fig, axes = plt.subplots(fig_n_rows, fig_n_cols, 
                                 figsize=(5.5 * fig_n_cols, 5.5 * fig_n_rows),
                                 squeeze=False)

        for i, model_name in enumerate(model_names):
            fold_tuples = model_conf_preds_folds[model_name]
            if j >= len(fold_tuples): continue
            
            (side_m1, y_truth, probs_m2, threshold, significance) = fold_tuples[j]
            
            # Convert to numpy arrays if they're pandas Series
            if hasattr(side_m1, 'values'): side_m1 = side_m1.values
            if hasattr(y_truth, 'values'): y_truth = y_truth.values
            if hasattr(probs_m2, 'values'): probs_m2 = probs_m2.values
            
            # Filter for non-zero truth if any exist (though usually binary in this project)
            valid_mask = (y_truth != 0)
            side_m1 = side_m1[valid_mask]
            y_truth = y_truth[valid_mask]
            probs_m2 = probs_m2[valid_mask]

            # Subplot Selection based on layout
            if rows_are_models:
                ax_a = axes[i, 0]
                ax_b = axes[i, 1]
                ax_c = axes[i, 2]
            else:
                ax_a = axes[0, i * 3]
                ax_b = axes[0, i * 3 + 1]
                ax_c = axes[0, i * 3 + 2]
            
            # --- PANEL A: M1 Confusion Matrix (2x2) ---
            cm_m1 = confusion_matrix(y_truth, side_m1, labels=[-1, 1])
            sns.heatmap(cm_m1, annot=True, fmt='d', cmap='Blues', ax=ax_a, cbar=False,
                        xticklabels=['Short', 'Long'], 
                        yticklabels=['Short', 'Long'],
                        annot_kws={'size': annot_fontsize})
            add_colorbar(ax_a.collections[0], ax_a)
            ax_a.tick_params(axis='both', labelsize=tick_font)
            
            # --- PANEL B: M2 Confusion (Can M2 predict M1 mistakes?) ---
            m1_correct = (side_m1 == y_truth).astype(int)
            m2_pred = (probs_m2 >= threshold).astype(int)
            cm_m2 = confusion_matrix(m1_correct, m2_pred, labels=[0, 1])
            sns.heatmap(cm_m2, annot=True, fmt='d', cmap='Reds', ax=ax_b, cbar=False,
                        xticklabels=['Wrong', 'Correct'],
                        yticklabels=['Wrong', 'Correct'],
                        annot_kws={'size': annot_fontsize})
            add_colorbar(ax_b.collections[0], ax_b)
            ax_b.tick_params(axis='both', labelsize=tick_font)
            
            # --- PANEL C: Final Confusion (Accepted signals only) ---
            accept_mask = (probs_m2 >= threshold)
            y_truth_acc = y_truth[accept_mask]
            side_m1_acc = side_m1[accept_mask]
            
            if len(y_truth_acc) > 0:
                cm_final = confusion_matrix(y_truth_acc, side_m1_acc, labels=[-1, 1])
                sns.heatmap(cm_final, annot=True, fmt='d', cmap='Greens', ax=ax_c, cbar=False,
                            xticklabels=['Short', 'Long'],
                            yticklabels=['Short', 'Long'],
                            annot_kws={'size': annot_fontsize})
                add_colorbar(ax_c.collections[0], ax_c)
                ax_c.tick_params(axis='both', labelsize=tick_font)
            else:
                ax_c.text(0.5, 0.5, "No Accepted Signals", ha='center', va='center')
                ax_c.tick_params(axis='both', labelsize=tick_font)
            
            # Metrics
            m1_errors = (side_m1 != y_truth).sum()
            final_errors = (side_m1_acc != y_truth_acc).sum() if len(y_truth_acc) > 0 else 0
            error_reduction = (m1_errors - final_errors) / m1_errors * 100 if m1_errors > 0 else 0
            
            # --- TITLES & LABELS ---
            # Group titles (Column headers)
            if rows_are_models:
                if i == 0:
                    ax_b.set_title(f'FOLD {j+1}', fontsize=ax_title_font, fontweight='bold', pad=35)
                    ax_a.set_title("Panel A: M1 Signals", fontsize=ax_title_font-4, color='dimgrey', pad=10)
                    ax_b.text(0.5, 1.05, "Panel B: M2 Meta-Model", transform=ax_b.transAxes, ha='center', fontsize=ax_title_font-4, color='dimgrey')
                    ax_c.set_title("Panel C: Final (Filtered)", fontsize=ax_title_font-4, color='dimgrey', pad=10)
            else:
                ax_b.set_title(f'{model_name}', fontsize=ax_title_font, fontweight='bold', pad=35)
                ax_a.set_title("Panel A: M1 Signals", fontsize=ax_title_font-4, color='dimgrey', pad=10)
                ax_b.text(0.5, 1.05, "Panel B: M2 Meta-Model", transform=ax_b.transAxes, ha='center', fontsize=ax_title_font-4, color='dimgrey')
                ax_c.set_title("Panel C: Final (Filtered)", fontsize=ax_title_font-4, color='dimgrey', pad=10)
            
            # Row labels
            if rows_are_models:
                ax_a.set_ylabel(f'{model_name}\nActual', fontsize=y_label_font, fontweight='bold')
            else:
                ax_a.set_ylabel(f'FOLD {j+1}\nActual', fontsize=y_label_font, fontweight='bold')
            
            ax_a.set_xlabel('Predicted', fontsize=x_label_font)
            ax_b.set_xlabel('M2 Prediction', fontsize=x_label_font)
            ax_b.set_ylabel('Actual (M1 Correct?)', fontsize=y_label_font)
            ax_c.set_xlabel('Predicted', fontsize=x_label_font)
            ax_c.set_ylabel('Actual', fontsize=y_label_font)
            
            # Stats for Panel C
            stats_txt = f'Rejected: {(~accept_mask).sum()}\nErrors -{error_reduction:.1f}%'
            ax_c.text(0.98, 0.02, stats_txt, transform=ax_c.transAxes, fontsize=stats_font, ha='right', va='bottom',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='lightgrey'))

        plt.suptitle('Nested Confusion Matrices: M1 → M2 → Conformal Filtering\n'
                    'M1: Base Signals | M2: Meta-Model Judgment | Final: Accepted Signals Only',
                    fontsize=20, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, bottom_margin, 1, 1-top_margin])
        
        os.makedirs("data/figures/confusion_matrices", exist_ok=True)
        if n_folds > 1:
            save_path = f"./data/figures/confusion_matrices/{title_pref}_nested_confusion_matrices_part{j+1}.png"
        else:
            save_path = f"./data/figures/confusion_matrices/{title_pref}_nested_confusion_matrices.png"
            
        plt.savefig(save_path, dpi=250, bbox_inches='tight')
        plt.close(fig)
        print(f"   ... Binary nested confusion matrices plot saved to {save_path}")


def generate_performance_table(granular_dfs: List[pd.DataFrame], model_name: str, phase_name: str = 'tournament'):
    """
    Generates a professional LaTeX table for inclusion in the thesis.
    Stacks results from multiple test folds (Outer Folds) and adds Average, 
    StDev, and an empty Interpretation column for manual entry.
    """
    if not granular_dfs:
        return

    # 1. Combine all folds into a MultiIndex DataFrame
    fold_keys = [f"Fold {i+1}" for i in range(len(granular_dfs))]
    full_df = pd.concat(granular_dfs, keys=fold_keys)
    full_df.index.names = ['Fold', 'Pair']

    # 2. Reshape to Metric-centric view
    reshaped_list = []
    for fold in fold_keys:
        fold_data = full_df.xs(fold, level='Fold').T
        fold_data.index.name = 'Metric'
        reshaped_list.append(fold_data)
    
    combined_metrics = pd.concat(reshaped_list, keys=fold_keys)
    combined_metrics.index.names = ['Fold', 'Metric']

    # 3. Add Average and StDev columns
    combined_metrics['Average'] = combined_metrics.mean(axis=1)
    combined_metrics['StDev'] = combined_metrics.std(axis=1)

    # 4. Add Empty Interpretation Column
    combined_metrics['Interpretation'] = ""

    # 5. Clean up for LaTeX
    final_table = combined_metrics.reset_index()

    # Replace underscores with spaces for LaTeX compatibility
    final_table['Metric'] = final_table['Metric'].str.replace('_', ' ', regex=False)
    final_table['Fold'] = final_table['Fold'].str.replace('_', ' ', regex=False)
    
    # Clear duplicate Fold labels
    final_table['Fold'] = final_table['Fold'].mask(final_table['Fold'].duplicated(), "")

    # Format numbers
    def format_val(x):
        if isinstance(x, (int, float)):
            if abs(x) < 0.01: return f"{x:.4f}"
            return f"{x:.2f}"
        return x

    formatted_table = final_table.map(format_val)

    # 6. Export to LaTeX
    os.makedirs(f"data/tables/{phase_name}/performance_tables", exist_ok=True)
    filename = f"data/tables/{phase_name}/performance_tables/perf_{model_name}_{phase_name}.tex"
    
    # Clean model name and phase for caption
    safe_model = model_name.replace('_', ' ')
    safe_phase = phase_name.replace('_', ' ').capitalize()
    
    # Use double braces in caption as per user's working fix
    caption = f"{{Granular Performance Analysis {safe_model} ({safe_phase} Phase)}}"
    label = f"tab:perf_{model_name}_{phase_name}"
    
    # Generate the base LaTeX code using sidewaystable
    latex_code = formatted_table.style.hide(axis='index').to_latex(
        caption=caption,
        label=label,
        environment="sidewaystable",
        position_float="centering",
        hrules=True
    )

    # --- IMPLEMENT "FIT TO WIDTH" ---
    latex_code = latex_code.replace(r'\begin{tabular}', r'\resizebox{\textwidth}{!}{\begin{tabular}')
    latex_code = latex_code.replace(r'\end{tabular}', r'\end{tabular}}')

    with open(filename, 'w') as f:
        f.write(latex_code)
    
    print(f"   ... Professional performance table (Rotated & Scaled) saved to: {filename}")


def generate_trading_hps_table(t_hps_list: List[dict], model_name: str, phase_name: str = 'tournament'):
    """
    Generates a professional LaTeX table showing the optimized trading 
    hyperparameters for each currency pair across multiple folds.
    """
    if not t_hps_list:
        return

    all_fold_dfs = []
    for i, t_hps in enumerate(t_hps_list):
        # 1. Parse the flat dictionary into a structured format
        parsed_data = {}
        for key, val in t_hps.items():
            suffixes = ['_sl_mult', '_tp_mult', '_risk_pct', '_kelly_fraction', '_mh', 
                        '_max_notional_exposure_pct', '_meta_significance']
            
            for suffix in suffixes:
                if key.endswith(suffix):
                    pair = key.replace(suffix, '')
                    param = suffix.lstrip('_')
                    
                    if pair not in parsed_data:
                        parsed_data[pair] = {}
                    parsed_data[pair][param] = val
                    break
        
        if parsed_data:
            fold_df = pd.DataFrame.from_dict(parsed_data, orient='index')
            fold_df.index.name = 'Pair'
            all_fold_dfs.append(fold_df)

    if not all_fold_dfs:
        print("!!! Warning: No per-pair trading HPs found to report.")
        return

    # 2. Combine all folds
    fold_keys = [f"Fold {i+1}" for i in range(len(all_fold_dfs))]
    full_df = pd.concat(all_fold_dfs, keys=fold_keys)
    full_df.index.names = ['Fold', 'Pair']
    
    # 3. Clean up for LaTeX
    final_table = full_df.reset_index()
    
    # Replace underscores in column names and values
    final_table.columns = [c.replace('_', ' ') for c in final_table.columns]
    
    # Clear duplicate Fold labels
    final_table['Fold'] = final_table['Fold'].str.replace('_', ' ', regex=False)
    final_table['Fold'] = final_table['Fold'].mask(final_table['Fold'].duplicated(), "")
    
    # Add empty Interpretation column
    final_table['Interpretation'] = ""

    # Format numbers
    def format_val(x):
        if isinstance(x, (int, float)):
            if abs(x) < 0.01: return f"{x:.4f}"
            return f"{x:.2f}"
        return x

    formatted_table = final_table.map(format_val)

    # 4. Export to LaTeX (Rotated & Scaled)
    os.makedirs(f"data/tables/{phase_name}/trading_hps_table", exist_ok=True)
    filename = f"data/tables/{phase_name}/trading_hps_table/hps_{model_name}_{phase_name}.tex"
    
    safe_model = model_name.replace('_', ' ')
    safe_phase = phase_name.replace('_', ' ').capitalize()
    caption = f"{{Optimized Per-Pair Trading Hyperparameters {safe_model} ({safe_phase} Phase)}}"
    label = f"tab:hps_{model_name}_{phase_name}"
    
    latex_code = formatted_table.style.hide(axis='index').to_latex(
        caption=caption,
        label=label,
        environment="sidewaystable",
        position_float="centering",
        hrules=True
    )

    # Apply "Fit to Width" scaling
    latex_code = latex_code.replace(r'\begin{tabular}', r'\resizebox{\textwidth}{!}{\begin{tabular}')
    latex_code = latex_code.replace(r'\end{tabular}', r'\end{tabular}}')

    with open(filename, 'w') as f:
        f.write(latex_code)
    
    print(f"   ... Professional Trading HPs table saved to: {filename}")


def generate_strategy_comparison_table(comparison_data: dict, phase_name: str = 'tournament'):
    """
    Generates professional LaTeX tables comparing different trading strategies
    (B&H, SMA, Simple ML, Complex) across folds and pairs for EACH model.
    
    Parameters:
    - comparison_data: dict { model_name: [list of per-pair DataFrames with strategy sharpes] }
    """
    if not comparison_data:
        return

    os.makedirs(f"data/tables/{phase_name}/strategy_comparison", exist_ok=True)

    for model_name, folds in comparison_data.items():
        all_rows = []
        for f_idx, fold_df in enumerate(folds):
            fold_label = f"Fold {f_idx + 1}"
            
            # Add results for each pair in this fold
            for pair, row in fold_df.iterrows():
                new_row = row.to_dict()
                new_row['Fold'] = fold_label
                new_row['Pair'] = pair
                all_rows.append(new_row)
            
            # Calculate Mean and Std for the "Fold Average" row
            means = fold_df.mean(numeric_only=True)
            stds = fold_df.std(numeric_only=True)
            
            # Use LaTeX bolding and plus-minus symbol
            avg_row = {'Fold': fold_label, 'Pair': r'\textbf{AVERAGE}'}
            for col in fold_df.columns:
                if col in means.index:
                    m = means[col]
                    s = stds[col]
                    # We wrap the entire cell in \textbf and use \pm in math mode
                    avg_row[col] = f"\\textbf{{{m:.2f} ($\\pm$ {s:.2f})}}"
            all_rows.append(avg_row)

        df = pd.DataFrame(all_rows)

        # Reorder columns
        cols = ['Fold', 'Pair', 'Buy and Hold', 'SMA Crossover', 'M1 Only', 'M1 + M2 (Fixed)', 'M1 + M2 (Global)', 'M1 + M2 (Conformal)']
        available_cols = [c for c in cols if c in df.columns]
        df = df[available_cols]

        # 4. Clean up for LaTeX
        # Fold changed if Fold name is different
        fold_changed = (df['Fold'] != df['Fold'].shift(1))

        df['Fold'] = df['Fold'].str.replace('_', ' ', regex=False)
        
        # Apply masks: Keep value ONLY where changed, else empty string
        df['Fold'] = df['Fold'].where(fold_changed, "")

        # Format numbers (only if they aren't already strings like the Average row)
        def format_val(x):
            if isinstance(x, (int, float)):
                return f"{x:.2f}"
            return str(x)

        # 5. Export to LaTeX (Upright & Scaled)
        filename = f"data/tables/{phase_name}/strategy_comparison/strategy_comp_{model_name}_{phase_name}.tex"
        
        safe_model = model_name.replace('_', ' ')
        safe_phase = phase_name.capitalize()
        caption = f"{{Cross-Strategy Performance Comparison: {safe_model} ({safe_phase} Phase) - Annualized Sharpe Ratio}}"
        label = f"tab:strategy_comparison_{model_name}_{phase_name}"
        
        # We no longer use .style.apply(bold_avg) as we manually bolded the cells
        latex_code = df.style.format(format_val).hide(axis='index').to_latex(
            caption=caption,
            label=label,
            position="H",
            position_float="centering",
            hrules=True
        )

        # Apply "Fit to Width" scaling
        latex_code = latex_code.replace(r'\begin{tabular}', r'\resizebox{\textwidth}{!}{\begin{tabular}')
        latex_code = latex_code.replace(r'\end{tabular}', r'\end{tabular}}')

        # Visually differentiate the AVERAGE row with bars
        lines = latex_code.split('\n')
        new_lines = []
        for line in lines:
            if r"\textbf{AVERAGE}" in line:
                # Add bars before and after the row to make it stand out
                new_lines.append(r"\midrule")
                new_lines.append(line)
                new_lines.append(r"\midrule")
            else:
                new_lines.append(line)
        latex_code = "\n".join(new_lines)

        with open(filename, 'w') as f:
            f.write(latex_code)
        
        print(f"   ... Professional Strategy Comparison table for {model_name} saved to: {filename}")

def generate_rrf_leaderboard(consolidated_results: Dict[str, List[pd.DataFrame]], 
                             phase_name: str = "tournament", 
                             top_n: int = 2):
    """
    Consolidates per-pair metrics across folds and identifies the best/worst 
    performing pairs using Reciprocal Rank Fusion (RRF).
    
    This function generates two tables:
    1. A summary table listing the Top and Bottom pairs per fold.
    2. A granular matrix showing the absolute Rank and RRF score for every pair.
    """
    if not consolidated_results:
        return

    # Metrics to include in RRF (higher is better for all of these)
    # max_dd is expected to be negative, so -0.05 (better) is > -0.15 (worse)
    rrf_metrics = ["total_return", "sharpe", "probabilistic_sharpe", 
                   "deflated_sharpe", "max_dd", "win_rate", "profit_factor", "cagr"]
    
    summary_rows = []
    matrix_rows = []
    k = 60  # RRF constant as per Cormack et al. (2009)

    # 1. Process each model and its list of fold DataFrames
    for m_name, folds in consolidated_results.items():
        for f_idx, df_fold in enumerate(folds):
            # Ensure we only use metrics present in the DataFrame
            valid_metrics = [m for m in rrf_metrics if m in df_fold.columns]
            
            if not valid_metrics:
                continue

            # Calculate RRF scores for this specific fold
            # rank(ascending=False) ensures highest values get rank 1
            ranks = df_fold[valid_metrics].rank(ascending=False, method="min")
            rrf_scores = (1.0 / (k + ranks)).sum(axis=1)
            
            # Create final rankings based on the fused scores
            final_ranks = rrf_scores.rank(ascending=False, method="min").astype(int)
            
            # Build Matrix Row (Rank + Score in brackets)
            m_row = {"Model": m_name, "Fold": f"Fold {f_idx+1}"}
            for pair in df_fold.index:
                score = rrf_scores.loc[pair]
                rank = final_ranks.loc[pair]
                m_row[pair] = f"{rank} ({score:.3f})"
            matrix_rows.append(m_row)
            
            # Build Summary Row (Best/Worst N pairs)
            sorted_pairs = rrf_scores.sort_values(ascending=False).index.tolist()
            summary_rows.append({
                "Model": m_name,
                "Fold": f"Fold {f_idx+1}",
                "Best Performers (RRF)": ", ".join(sorted_pairs[:top_n]),
                "Worst Performers (RRF)": ", ".join(sorted_pairs[-top_n:][::-1])
            })

    # 2. Convert to DataFrames and format for LaTeX
    df_summary = pd.DataFrame(summary_rows)
    df_matrix = pd.DataFrame(matrix_rows)

    # Mask repeating Model/Fold names for visual clarity
    for df in [df_summary, df_matrix]:
        model_changed = df["Model"] != df["Model"].shift(1)
        df["Model"] = df["Model"].where(model_changed, "")
        df["Fold"] = df["Fold"].where(model_changed | (df["Fold"] != df["Fold"].shift(1)), "")

    os.makedirs(f"data/tables/{phase_name}/rrf_leaderboard", exist_ok=True)
    
    # --- TABLE 1: SUMMARY ---
    summary_path = f"data/tables/{phase_name}/rrf_leaderboard/rrf_summary_{phase_name}.tex"
    summary_caption = f"{{RRF Performance Summary: Best and Worst Performing Pairs ({phase_name.capitalize()} Phase)}}"
    summary_label = f"tab:rrf_summary_{phase_name}"
    
    # Use Styler for more robust LaTeX export
    summary_tex = df_summary.style.hide(axis="index").to_latex(
        caption=summary_caption,
        label=summary_label,
        position="H",
        column_format="l l p{5cm} p{5cm}",
        hrules=True
    )
    
    with open(summary_path, "w") as f:
        f.write(summary_tex)

    # --- TABLE 2: GRANULAR MATRIX ---
    matrix_path = f"data/tables/{phase_name}/rrf_leaderboard/rrf_matrix_{phase_name}.tex"
    matrix_caption = f"{{RRF Granular rankings and Fused Scores ({phase_name.capitalize()} Phase)}}"
    matrix_label = f"tab:rrf_matrix_{phase_name}"
    
    # Generate LaTeX with sidewaystable environment for width
    matrix_tex = df_matrix.style.hide(axis="index").to_latex(
        environment="sidewaystable",
        caption=matrix_caption,
        label=matrix_label,
        hrules=True
    )
    
    # Apply Resizebox scaling to fit text width
    matrix_tex = matrix_tex.replace(r"\begin{tabular}", r"\resizebox{\textwidth}{!}{\begin{tabular}")
    matrix_tex = matrix_tex.replace(r"\end{tabular}", r"\end{tabular}}")
    
    with open(matrix_path, "w") as f:
        f.write(matrix_tex)
    
    print(f"   ... RRF Summary and Matrix tables saved to data/tables/{phase_name}/rrf_leaderboard")


def plot_nested_reliability_diagrams(model_conf_preds_folds, n_outer_splits, title_pref, rows_are_models=True):
    """
    Creates Reliability Diagrams (Calibration Curves) for the M2 Meta-Model.
    
    This plot assesses how well M2's predicted probabilities correspond to real outcomes.
    If M2 says 80% confidence, is M1 actually correct 80% of the time?
    
    Layout: 1 Panel per Model/Fold
    - Main Plot: Observed Accuracy vs. Predicted Confidence (10 bins)
    - Reference: 45-degree 'Perfectly Calibrated' line
    - Inset/Text: Brier Score (Lower is better, 0.0 is perfect)
    - Sub-panel: Distribution histogram of predicted probabilities
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    
    model_names = list(model_conf_preds_folds.keys())
    n_models = len(model_names)
    n_folds = n_outer_splits

    # Chunking Logic: If rows_are_models, we limit to 3 models per figure
    if rows_are_models:
        chunk_size = 3
        model_chunks = [model_names[i:i + chunk_size] for i in range(0, n_models, chunk_size)]
    else:
        model_chunks = [model_names]

    for part_idx, current_model_names in enumerate(model_chunks):
        n_current_models = len(current_model_names)
        
        if rows_are_models:
            n_rows, n_cols = n_current_models, n_folds
            legend_font = 20
            y_label_font = 20
            ax_title_font = 20
            x_label_font = 20
            top_margin = 0.02
            stats_font = 20
            if n_current_models == 3:
                bottom_margin = 0.03
                legend_y = -0.01
            elif n_current_models < 3:
                bottom_margin = 0.055
                legend_y = -0.045
        else:
            n_rows, n_cols = n_folds, n_models
            legend_font = 16
            legend_y = -0.045
            y_label_font = 16
            ax_title_font = 16
            x_label_font = 14
            bottom_margin = 0.03
            top_margin = 0.05
            stats_font = 10
            
        fig, axes = plt.subplots(n_rows, n_cols, 
                                 figsize=(6 * n_cols, 6 * n_rows),
                                 squeeze=False)
        
        for i, model_name in enumerate(current_model_names):
            fold_tuples = model_conf_preds_folds[model_name]
            
            for j, (side_m1, y_truth, probs_m2, threshold, significance) in enumerate(fold_tuples):
                
                # 1. Prepare Data
                if hasattr(side_m1, 'values'): side_m1 = side_m1.values
                if hasattr(y_truth, 'values'): y_truth = y_truth.values
                if hasattr(probs_m2, 'values'): probs_m2 = probs_m2.values
                
                # Ground Truth for M2: Was M1 Directionally Correct?
                y_true_m2 = (side_m1 == y_truth).astype(int)
                
                # 2. Calculate Calibration Curve
                prob_true, prob_pred = calibration_curve(y_true_m2, probs_m2, n_bins=10, strategy='uniform')
                
                # 3. Calculate Quantitative Metrics
                brier = brier_score_loss(y_true_m2, probs_m2)
                
                # Subplot Selection
                ax = axes[i, j] if rows_are_models else axes[j, i]
                
                # --- MAIN CALIBRATION PLOT ---
                ax.plot([0, 1], [0, 1], linestyle='--', color='grey', alpha=0.6, label='Perfectly Calibrated')
                ax.plot(prob_pred, prob_true, marker='s', markersize=4, linewidth=2, 
                        color='#3498db', label=f'M2: {model_name}')
                ax.fill_between(prob_pred, prob_pred, prob_true, color='#3498db', alpha=0.1)
                
                # --- PROBABILITY DISTRIBUTION (Histogram Overlay) ---
                ax_hist = ax.twinx()
                ax_hist.hist(probs_m2, bins=10, range=(0, 1), alpha=0.15, color='grey', edgecolor='none')
                ax_hist.set_ylim(0, len(probs_m2))
                ax_hist.axis('off')
                
                # --- FORMATTING & ANNOTATION ---
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks(np.arange(0, 1.1, 0.1))
                ax.set_yticks(np.arange(0, 1.1, 0.1))
                ax.grid(True, alpha=0.2, linestyle=':')
                
                stats_box = f"Brier Score: {brier:.4f}"
                ax.text(0.05, 0.95, stats_box, transform=ax.transAxes, verticalalignment='top',
                        fontsize=stats_font, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Logic for Titles and Labels
                if rows_are_models:
                    if j == 0: ax.set_ylabel(f"{model_name}\nFraction of Positives", fontsize=y_label_font, fontweight='bold')
                    if i == 0: ax.set_title(f"TEST FOLD {j+1}", fontsize=ax_title_font, fontweight='bold', pad=15)
                    if i == n_current_models - 1: ax.set_xlabel("Mean Predicted Confidence", fontsize=x_label_font)
                else:
                    if i == 0: ax.set_ylabel(f"GLOBAL TEST FOLD {j+1}\nFraction of Positives", fontsize=y_label_font, fontweight='bold')
                    if j == 0: ax.set_title(f"{model_name}", fontsize=ax_title_font, fontweight='bold', pad=15)
                    if j == n_folds - 1: ax.set_xlabel("Mean Predicted Confidence", fontsize=x_label_font)

        fig.suptitle(f"Nested Reliability Analysis: M2 Confidence Calibration ({title_pref})", 
                     fontsize=22, fontweight='bold', y=0.98)
        
        # Global Legend
        legend_elements = [
            Line2D([0], [0], color='grey', linestyle='--', label='Theoretical Perfection'),
            Line2D([0], [0], color='#3498db', marker='s', label='Observed Calibration'),
            Line2D([0], [0], color='grey', alpha=0.2, label='Confidence Density (Samples)')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3, frameon=True, 
                   framealpha=1.0, fontsize=legend_font, edgecolor='black', bbox_to_anchor=(0.5, legend_y))

        plt.tight_layout(rect=[0, bottom_margin, 1, 1-top_margin])
        
        os.makedirs("data/figures/reliability_diagrams", exist_ok=True)
        
        if rows_are_models:
            suffix = f"_part{part_idx + 1}"
        else:
            suffix = ""
            
        save_path = f"./data/figures/reliability_diagrams/{title_pref}_reliability_diagrams{suffix}.png"
        plt.savefig(save_path, dpi=250, bbox_inches='tight')
        plt.close(fig)
        print(f"   ... Reliability diagrams saved to {save_path}")


def generate_tournament_summary_tables(consolidated_results: Dict[str, List[pd.DataFrame]], 
                                       phase_name: str = 'tournament'):
    """
    Generates high-level summary tables for the thesis:
    1. tab:all_models: Detailed Statistics by Model (mean ± std format)
    2. tab:ranking: Best performer per metric + Overall RRF Winner (Global Means)
    """
    if not consolidated_results:
        return

    # 1. AGGREGATE DATA
    model_stats = {}
    model_means = {} # For RRF calculation

    # Standard metric metadata
    metric_meta = {
        'total_return': ('Total Return (\\%)', True),
        'sharpe': ('Sharpe Ratio', True),
        'probabilistic_sharpe': ('Probabilistic Sharpe (\\%)', True),
        'max_dd': ('Max Drawdown (\\%)', True),
        'cagr': ('CAGR (\\%)', True),
        'win_rate': ('Win Rate (\\%)', True),
        'profit_factor': ('Profit Factor', True),
        'n_trades': ('Number of Trades', False),
        'avg_capital_exposure': ('Avg Capital Exposure', False),
        'avg_trade_size': ('Avg Trade Size', False),
        'deflated_sharpe': ('Deflated Sharpe', True),
        'pfdr': ('PFDR', False),
        'm2_brier': ('M2 Brier', False)
    }

    for m_name, folds in consolidated_results.items():
        if not folds: continue
        # Pool all pairs across all folds for this model
        df_full = pd.concat(folds)

        means = df_full.mean(numeric_only=True)
        stds = df_full.std(numeric_only=True)

        model_means[m_name] = means

        # Build "mean ± std" formatted strings
        formatted_summary = {}
        for key, (display_name, _) in metric_meta.items():
            if key in means.index:
                m = means[key]
                s = stds[key]

                # Check if it's a percentage metric
                if "(%)" in display_name:
                    formatted_summary[display_name] = f"${m*100:.2f}\\% \\pm {s*100:.2f}\\%$"
                else:
                    formatted_summary[display_name] = f"${m:.4f} \\pm {s:.4f}$"

        model_stats[m_name] = formatted_summary

    # 2. CREATE TABLE 1: Detailed Statistics
    df_all_models = pd.DataFrame(model_stats)
    df_all_models.index.name = 'Statistic'
    df_all_models = df_all_models.reset_index()

    # 3. CREATE TABLE 2: Ranking Summary + RRF
    df_means_matrix = pd.DataFrame(model_means)
    ranking_rows = []

    # Calculate RRF Scores based on Global Means
    k_rrf = 60
    # Higher is better metrics
    hb_metrics = [k for k, v in metric_meta.items() if v[1]]
    # Lower is better metrics
    lb_metrics = [k for k, v in metric_meta.items() if not v[1]]

    # Ranks for RRF
    ranks_hb = df_means_matrix.loc[df_means_matrix.index.intersection(hb_metrics)].rank(ascending=False, axis=1)
    ranks_lb = df_means_matrix.loc[df_means_matrix.index.intersection(lb_metrics)].rank(ascending=True, axis=1)
    all_ranks = pd.concat([ranks_hb, ranks_lb])

    rrf_scores = (1.0 / (k_rrf + all_ranks)).sum(axis=0)
    rrf_winner = rrf_scores.idxmax()
    rrf_winner_score = rrf_scores.max()

    # Build the metric-by-metric ranking rows
    for key, (display_name, higher_is_better) in metric_meta.items():
        if key not in df_means_matrix.index: continue

        vals = df_means_matrix.loc[key]
        best_model = vals.idxmax() if higher_is_better else vals.idxmin()
        best_val_raw = vals.max() if higher_is_better else vals.min()

        # Format the best value in math mode
        if "(%)" in display_name:
            best_val = f"${best_val_raw*100:.2f}\\%$"
        else:
            best_val = f"${best_val_raw:.4f}$"

        ranking_rows.append({
            'Metric': display_name,
            'Higher is Better': 'Yes' if higher_is_better else 'No',
            'Best Model': best_model,
            'Best Value': best_val
        })

    df_ranking = pd.DataFrame(ranking_rows)

    # 4. EXPORT TO LATEX
    os.makedirs(f"data/tables/{phase_name}/summary_tables", exist_ok=True)

    # Clean phase name for display in captions
    display_phase = phase_name.replace('_', ' ').capitalize()

    # --- tab:all_models ---
    path_all = f"data/tables/{phase_name}/summary_tables/all_models_{phase_name}.tex"
    caption_all = f"{{Detailed Statistics by Model (Mean $\\pm$ Std) - {display_phase} Phase}}"
    label_all = f"tab:all_models_{phase_name}"

    tex_all = df_all_models.style.hide(axis='index').to_latex(
        caption=caption_all, label=label_all, position="H", position_float="centering", hrules=True
    )
    # Fit to width
    tex_all = tex_all.replace(r'\begin{tabular}', r'\resizebox{\textwidth}{!}{\begin{tabular}')
    tex_all = tex_all.replace(r'\end{tabular}', r'\end{tabular}}')

    with open(path_all, 'w') as f:
        f.write(tex_all)

    # --- tab:ranking ---
    path_rank = f"data/tables/{phase_name}/summary_tables/ranking_{phase_name}.tex"
    caption_rank = f"{{Ranking Summary - Best Performer per Metric ({display_phase} Phase)}}"
    label_rank = f"tab:ranking_{phase_name}"
    # Generate base ranking table
    tex_rank = df_ranking.style.hide(axis='index').to_latex(
        caption=caption_rank, label=label_rank, position="H", position_float="centering", hrules=True
    )

    # Insert the RRF winner row before the end of the tabular
    rrf_row = f"\\\\ \\midrule \\textbf{{Overall RRF Winner}} & & \\textbf{{{rrf_winner}}} & \\textbf{{Score: {rrf_winner_score:.4f}}} \\\\"
    tex_rank = tex_rank.replace(r'\bottomrule', rrf_row + r'\bottomrule')

    # Fit to width
    tex_rank = tex_rank.replace(r'\begin{tabular}', r'\resizebox{\textwidth}{!}{\begin{tabular}')
    tex_rank = tex_rank.replace(r'\end{tabular}', r'\end{tabular}}')

    with open(path_rank, 'w') as f:
        f.write(tex_rank)

    print(f"   ... Tournament Summary tables generated in data/tables/{phase_name}/summary_tables/")