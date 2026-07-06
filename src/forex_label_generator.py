"""
Forex Label Generation Module
==============================

This module implements the labeling logic for the Meta-Labeling strategy.
It generates two sets of labels:
1. M1 Labels (Binary Side): Pure directional bets based on a future horizon.
2. Ground Truth (Triple Barrier): The actual market outcome based on volatility-adjusted
   take-profit, stop-loss, and time-exhaustion barriers (Marcos Lopez de Prado).

Institutional Standard:
- Robust ATR-based volatility scaling for barriers.
- Efficient path-dependent labeling logic.
- Automated data alignment and NaN-purging for ML readiness.
"""

import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Union

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

FEATURES_DIR = "./data/csv_files/forex_features_raw"
COMPLETE_DIR = "./data/csv_files/forex_complete_data"

# ==============================================================================
#  LABEL DISTRIBUTION
# ==============================================================================

def count_label_dist():
    """
    Calculates and saves the distribution of labels (Long, Short, Timeout).
    Generates a professional LaTeX table for inclusion in the thesis document.
    """
    # 1. Directory Setup
    os.makedirs("./data/tables", exist_ok=True)
    
    filename = "aggregated_complete_data.csv"
    filepath = f"./data/csv_files/forex_master_data/{filename}"

    if not os.path.exists(filepath):
        print(f"\n!!! Warning: {filepath} not found.")
        print("    Ensure aggregate_forex_data() has been run.")
        return

    df = pd.read_csv(filepath)
    label = df['y_truth']
    total_len = len(label)

    # 2. Calculate Distribution
    long_count = np.sum(df['y_truth'] == 1)
    short_count = np.sum(df['y_truth'] == -1)
    timeout_count = np.sum(df['y_truth'] == 0)

    # 4. LaTeX Table Generation
    dist_df = pd.DataFrame({
        'Label': ['Short', 'Long', 'Timeout', 'Total'],
        'Count': [short_count, long_count, timeout_count, total_len],
        'Percentage': [
            f"{short_count / total_len * 100 :.2f}\%",
            f"{long_count / total_len * 100 :.2f}\%",
            f"{timeout_count / total_len * 100 :.2f}\%",
            "100.00\%"
        ]
    })

    tex_filename = f"data/tables/label_dist.tex"
    
    latex_code = dist_df.style.hide(axis='index').to_latex(
        caption="Distribution of Triple-Barrier Labels (Aggregated Dataset)",
        label="tab:label_dist",
        position="h",
        position_float="centering",
        hrules=True
    )

    with open(tex_filename, 'w') as f:
        f.write(latex_code)
    
    print(f"   ... LaTeX table saved to: {tex_filename}")
    print("="*40 + "\n")


# ==============================================================================
# 1. MAIN LABELING PIPELINE
# ==============================================================================

def generate_forex_labels(t_final: int, 
                          atr_lookback: int, 
                          tp_atr_multiplier: float, 
                          sl_atr_multiplier: float) -> None:
    """
    Orchestrates label generation for all available currency features.

    Args:
        horizon (int): The forward window (legacy, now t_final is used for both TBM labels).
        t_final (int): Maximum holding period for the Triple Barrier Method.
        atr_lookback (int): Window size for ATR volatility calculation.
        tp_atr_multiplier (float): Multiplier for the upper barrier.
        sl_atr_multiplier (float): Multiplier for the lower barrier.
    """
    os.makedirs(COMPLETE_DIR, exist_ok=True)
    files = glob.glob(f"{FEATURES_DIR}/*.csv")

    start_time = datetime.now()
    print("\n" + "="*80)
    print("FOREX LABEL GENERATOR (META-LABELING)")
    print(f"Started at: {start_time}")
    print(f"Config: T_FINAL={t_final}, ATR_LB={atr_lookback}")
    print("="*80)

    for i, file in enumerate(files, 1):
        pair_name = os.path.basename(file).replace("_features_raw.csv", "")
        print(f"\n[{i}/{len(files)}] Generating labels for: {pair_name}")
        print("-" * 40)
        
        # Load and align
        df = pd.read_csv(file, index_col="date")
        df.index = pd.to_datetime(df.index)

        # Prepare price data (naming normalization)
        prices = df[['tx_high', 'tx_low', 'tx_close']].copy()
        prices.columns = ['high', 'low', 'close']

        # Step 1: Generate Primary Model (M1) Directional Labels
        # NOW: Use Binary TBM (sign_on_timeout=True) to force a directional bet
        y_side = compute_triple_barrier_labels(
            prices=prices,
            t_final=t_final,
            atr_lookback=atr_lookback,
            tp_atr_multiplier=tp_atr_multiplier,
            sl_atr_multiplier=sl_atr_multiplier,
            sign_on_timeout=True
        )
        
        # Step 2: Generate Triple-Barrier Ground Truth (Referee)
        # For M2 to judge M1, we use the same TBM logic.
        y_truth = compute_triple_barrier_labels(
            prices=prices,
            t_final=t_final,
            atr_lookback=atr_lookback,
            tp_atr_multiplier=tp_atr_multiplier,
            sl_atr_multiplier=sl_atr_multiplier,
            sign_on_timeout=True
        )

        # Step 3: Consolidate (Defragmented insertion)
        df = pd.concat([df, y_side.rename('y_side'), y_truth.rename('y_truth')], axis=1)
        
        # Step 4: Purge calculation artifacts (NaNs from shifts/warmups)
        initial_len = len(df)
        df.dropna(inplace=True)
        final_len = len(df)
        
        # Persistence
        output_path = f"{COMPLETE_DIR}/{pair_name}_complete_data.csv"
        df.to_csv(output_path)
        
        print(f"   ... M1/Truth (Binary TBM) labels generated.")
        print(f"   ... Data points preserved: {final_len} (Purged {initial_len - final_len} NaNs)")

    duration = datetime.now() - start_time
    print("\n" + "="*80)
    print("LABEL GENERATION COMPLETE")
    print(f"Total time: {int(duration.total_seconds())}s")
    print("="*80)

# ==============================================================================
# 2. CORE LABELING LOGIC
# ==============================================================================

def compute_triple_barrier_labels(prices: pd.DataFrame, 
                                 t_final: int, 
                                 atr_lookback: int, 
                                 tp_atr_multiplier: float, 
                                 sl_atr_multiplier: float,
                                 sign_on_timeout: bool = True,
                                 only_tpsl_percentages: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """
    Implements the Triple-Barrier Method for labeling financial time series.

    Assigns labels based on which barrier is hit first:
    - 1: Profit barrier (Close + ATR * Multiplier)
    - -1: Loss barrier (Close - ATR * Multiplier)
    - 0: Time barrier (Holding period expired) -> If sign_on_timeout=False
    - 1/-1: Time barrier (Holding period expired) -> If sign_on_timeout=True (Sign of return)

    Args:
        prices: DataFrame with ['high', 'low', 'close'].
        t_final: Max bars to hold.
        atr_lookback: ATR window.
        tp_atr_multiplier: ATR multiplier for profit taking.
        sl_atr_multiplier: ATR multiplier for stop loss.
        sign_on_timeout: If True, timeouts are converted to 1 or -1 based on price action.
        only_tpsl_percentages: If True, returns the raw barrier distances.

    Returns:
        pd.Series of labels or Tuple of percentage series.
    """
    # 1. Volatility and Barrier Definition
    atr = _compute_atr(prices, lookback=atr_lookback)
    atr = atr.replace(0, np.nan).ffill() 
    
    tp_levels = prices['close'] + (atr * tp_atr_multiplier)
    sl_levels = prices['close'] - (atr * sl_atr_multiplier)
    
    if only_tpsl_percentages:
        tp_pct = ((tp_levels - prices['close']) / prices['close']) * 100
        sl_pct = abs(((prices['close'] - sl_levels) / prices['close']) * 100)
        return tp_pct, sl_pct
    
    # Initialize labels
    labels = pd.Series(0, index=prices.index, dtype=int)
    
    # Performance Optimization: Extract values for faster iteration
    p_high = prices['high'].values
    p_low = prices['low'].values
    p_close = prices['close'].values
    tp_v = tp_levels.values
    sl_v = sl_levels.values
    n = len(prices)

    # 2. Sequential Path Search
    # Note: We iterate until n - t_final to allow the path to complete
    for i in range(n - t_final):
        # Skip if barriers are NaN (warmup period)
        if np.isnan(tp_v[i]) or np.isnan(sl_v[i]):
            continue
            
        tp, sl = tp_v[i], sl_v[i]
        barrier_hit = False
        
        # Check future path window
        for j in range(1, t_final + 1):
            idx = i + j
            # Rule: SL hit has priority if both hit in the same bar
            if p_low[idx] <= sl:
                labels.iloc[i] = -1
                barrier_hit = True
                break
            if p_high[idx] >= tp:
                labels.iloc[i] = 1
                barrier_hit = True
                break
        
        # Handle Timeout
        if not barrier_hit and sign_on_timeout:
            # If no barrier hit, label based on the sign of the price move at t_final
            if p_close[i + t_final] > p_close[i]:
                labels.iloc[i] = 1
            else:
                labels.iloc[i] = -1 # Includes price flat as -1 to maintain binary
    
    return labels


# ==============================================================================
# 3. STATISTICAL HELPERS
# ==============================================================================

def _compute_atr(prices: pd.DataFrame, lookback: int) -> pd.Series:
    """
    Calculates the Average True Range (ATR).
    """
    h_l = prices['high'] - prices['low']
    h_pc = np.abs(prices['high'] - prices['close'].shift(1))
    l_pc = np.abs(prices['low'] - prices['close'].shift(1))
    
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.rolling(window=lookback, min_periods=lookback).mean()

if __name__ == "__main__":
    # Internal module testing
    generate_forex_labels(
        horizon=5,
        t_final=10,
        atr_lookback=14,
        tp_atr_multiplier=2.0,
        sl_atr_multiplier=2.0
    )
