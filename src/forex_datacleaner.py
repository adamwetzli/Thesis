"""
Forex Data Cleaning & Validation Module
========================================

This module provides an institutional-grade cleaning pipeline for combined forex data.
It performs sanity checks, identifies "impossible" market bars, and analyzes
statistical violations between transaction prices and bid/ask spreads.

Core Processes:
1. Column Standardizing: Maps various source columns to a unified schema.
2. Market Logic Validation: Removes bars where High < Low or OHLC values are out of bounds.
3. Timestamp Monotonicity: Ensures chronological ordering and identifies gaps.
4. Bid/Ask Violation Analysis: Detects transaction prices that fall outside the 
   prevailing bid/ask spread (with optional slippage buffers).
"""

import os
import re
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from tabulate import tabulate

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

COMBINED_DIR = "./data/csv_files/forex_combined"
CLEANED_DIR = "./data/csv_files/forex_cleaned_and_validated"
REPORT_DIR = "./data/txt_files/forex_clean_val_reports"

# Mapping source columns to unified internal naming
COLUMN_RENAME_MAP = {
    "open": "tx_open", "high": "tx_high", "low": "tx_low", "close": "tx_close",
    "volume": "tx_volume", "vwap": "tx_vwap", "transactions": "tx_transactions",
    "BID_open": "bid_open", "BID_high": "bid_high", "BID_low": "bid_low", "BID_close": "bid_close",
    "BID_volume_millions": "bid_volume_mil",
    "ASK_open": "ask_open", "ASK_high": "ask_high", "ASK_low": "ask_low", "ASK_close": "ask_close",
    "ASK_volume_millions": "ask_volume_mil"
}

# Standard pip multipliers for major and minor pairs
PIP_MULTIPLIERS = {
    "USDJPY": 100,
    "EURJPY": 100,
    "GBPJPY": 100,
    # Default for most other pairs is 10,000
}
DEFAULT_PIP_MULT = 10000

# ==============================================================================
# 1. MAIN CLEANING ENGINE
# ==============================================================================

def clean_raw_forex_data(incl_viol_out: bool = True) -> None:
    """
    Main orchestration function for the cleaning and validation pipeline.

    Args:
        incl_viol_out (bool): If True, performs expensive Bid/Ask violation analysis.
    """
    files = glob.glob(f"{COMBINED_DIR}/*.csv")
    os.makedirs(CLEANED_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    start_time = datetime.now()

    print("\n" + "="*80)
    print("FOREX DATA CLEANING & VALIDATION")
    print(f"Started at: {start_time}")
    print("="*80)

    for i, file in enumerate(files):
        pair_name = os.path.basename(file).replace('_combined_raw.csv', '')
        print(f"\n[{i+1}/{len(files)}] Processing: {pair_name}")
        print("-" * 40)

        # 1. Load and Standardize
        raw = pd.read_csv(file, index_col="date")
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        raw.rename(columns=COLUMN_RENAME_MAP, inplace=True)

        # 2. Hard Validation (Sanity Checks)
        try:
            impossibles = remove_impossible_bars(raw)
            check_and_fix_timestamps_monotonic(raw)
        except KeyboardInterrupt:
            print("\n!!! Interrupted by user. Exiting safely.")
            break

        # 3. Soft Validation (Market Violations)
        pip_mult = PIP_MULTIPLIERS.get(pair_name, DEFAULT_PIP_MULT)
        
        magnitudes_wo = pd.DataFrame()
        magnitudes_w = pd.DataFrame()

        if incl_viol_out:
            try:
                # Analysis Without Slippage
                enhanced_wo = get_bid_ask_tx_violations(raw, 
                                                       open_close_slippage=0,
                                                       high_low_slippage=0,
                                                       pip_multiplier=pip_mult)
                if not enhanced_wo.empty:
                    print("   ... Running violation analysis (No Slippage Buffer)")
                    magnitudes_wo, _ = analyze_violation_magnitudes(enhanced_wo)

                # Analysis With Slippage
                enhanced_w = get_bid_ask_tx_violations(raw, pip_multiplier=pip_mult)
                if not enhanced_w.empty:
                    print("   ... Running violation analysis (With Slippage Buffer)")
                    magnitudes_w, _ = analyze_violation_magnitudes(enhanced_w)
            except KeyboardInterrupt:
                break
        
        # 4. Reporting & Persistence
        report_data = {
            "Analysis of Violations Without Slippage Buffer": magnitudes_wo,
            "Analysis of Violations With Slippage Buffer": magnitudes_w,
            "Impossible Bars Removed": impossibles
        }
        
        report_path = f"{REPORT_DIR}/{pair_name}_report.txt"
        create_report(report_data, report_path)

        clean_path = f"{CLEANED_DIR}/{pair_name}_clean.csv"
        raw.to_csv(clean_path)
        print(f"   ... Processed {pair_name}. Cleaned data saved to {clean_path}")

    duration = datetime.now() - start_time
    print("\n" + "="*80)
    print("CLEANING COMPLETE")
    print(f"Total time: {str(duration).split('.')[0]}")
    print("="*80)

# ==============================================================================
# 2. VALIDATION UTILITIES
# ==============================================================================

def remove_impossible_bars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies and removes bars that violate fundamental market logic.
    """
    valid_mask = pd.Series(True, index=df.index)
    
    check_groups = [
        ("tx", ["tx_open", "tx_high", "tx_low", "tx_close"], "tx_volume"),
        ("bid", ["bid_open", "bid_high", "bid_low", "bid_close"], "bid_volume_mil"),
        ("ask", ["ask_open", "ask_high", "ask_low", "ask_close"], "ask_volume_mil")
    ]
    
    for prefix, ohlc, vol in check_groups:
        if not all(col in df.columns for col in ohlc):
            continue
            
        o, h, l, c = ohlc
        
        # Diagnostic sub-masks
        m_h_l = (df[h] >= df[l])
        m_o_range = (df[o] >= df[l]) & (df[o] <= df[h])
        m_c_range = (df[c] >= df[l]) & (df[c] <= df[h])
        m_pos = (df[h] > 0) & (df[l] > 0)
        
        group_mask = m_h_l & m_o_range & m_c_range & m_pos
        
        if vol in df.columns:
            group_mask &= (df[vol] >= 0)
        
        # Diagnostic printing
        if not group_mask.all():
            print(f"   ... {prefix.upper()} Validation issues:")
            if not m_h_l.all(): print(f"       - High < Low: {(~m_h_l).sum()} instances")
            if not m_o_range.all(): print(f"       - Open out of range: {(~m_o_range).sum()} instances")
            if not m_c_range.all(): print(f"       - Close out of range: {(~m_c_range).sum()} instances")
            if not m_pos.all(): print(f"       - Zero/Negative prices: {(~m_pos).sum()} instances")

        valid_mask &= group_mask
    
    impossibles = df[~valid_mask].copy()
    num_removed = len(impossibles)
    
    if num_removed > 0:
        print(f"   ... Total removed: {num_removed} impossible bars.")
        df.drop(df[~valid_mask].index, inplace=True)
    
    return impossibles

def check_and_fix_timestamps_monotonic(df: pd.DataFrame) -> None:
    """
    Ensures that the index is strictly monotonic increasing.

    Args:
        df (pd.DataFrame): The DataFrame to check (modified IN-PLACE).
    """
    if df.index.is_monotonic_increasing:
        return
    
    # Calculate non-monotonic count
    diffs = df.index.to_series().diff().dropna()
    non_monotonic = (diffs <= pd.Timedelta(0)).sum()
    
    print(f"   ... Warning: Found {non_monotonic} non-monotonic timestamps. Sorting index.")
    df.sort_index(inplace=True)

# ==============================================================================
# 3. STATISTICAL VIOLATION ANALYSIS
# ==============================================================================

def get_bid_ask_tx_violations(df: pd.DataFrame, 
                             open_close_slippage: float = 2.0,
                             high_low_slippage: float = 5.0,
                             pip_multiplier: float = 10000) -> pd.DataFrame:
    """
    Detects transaction prices that fall outside the Bid/Ask range.

    Args:
        df: Input DataFrame.
        open_close_slippage: Buffer allowed for open/close prices (in pips).
        high_low_slippage: Buffer allowed for high/low prices (in pips).
        pip_multiplier: Factor to convert price difference to pips.

    Returns:
        pd.DataFrame: A descriptive DataFrame containing all violations.
    """
    result_df = pd.DataFrame(index=df.index)
    any_violation = pd.Series(False, index=df.index)
    
    for pt in ['open', 'high', 'low', 'close']:
        bid, tx, ask = f'bid_{pt}', f'tx_{pt}', f'ask_{pt}'
        if not all(col in df.columns for col in [bid, tx, ask]):
            continue
        
        buffer = (open_close_slippage if pt in ['open', 'close'] else high_low_slippage) / pip_multiplier
        
        tx_below = df[tx] < (df[bid] - buffer)
        tx_above = df[tx] > (df[ask] + buffer)
        mask = tx_below | tx_above
        any_violation |= mask
        
        # Calculate magnitudes
        bid_diff = (df[bid] - df[tx]) * pip_multiplier
        ask_diff = (df[tx] - df[ask]) * pip_multiplier
        
        result_df[f'{pt}_bid_tx_diff_pips'] = bid_diff.where(mask)
        result_df[f'{pt}_tx_ask_diff_pips'] = ask_diff.where(mask)
        
        # Build descriptive strings using vectorized logic where possible
        # However, to keep semantic behavior exactly as original, we use a loop for the string building
        descriptions = pd.Series('', index=df.index)
        viol_indices = df.index[mask]
        for idx in viol_indices:
            if tx_below.loc[idx]:
                descriptions.loc[idx] = f"tx_{pt} {bid_diff.loc[idx]:.1f}pips below bid"
            else:
                descriptions.loc[idx] = f"tx_{pt} {ask_diff.loc[idx]:.1f}pips above ask"
        
        result_df[f'{pt}_violation_desc'] = descriptions.where(mask)
    
    if not any_violation.any():
        return pd.DataFrame()
    
    # Return context + violations
    enhanced = result_df[any_violation].copy()
    for col in df.columns:
        enhanced[col] = df.loc[enhanced.index, col]
        
    return enhanced

def analyze_violation_magnitudes(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Statistical analysis of the distribution of pip-violations.

    Args:
        df: Enhanced DataFrame containing violation descriptions.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (Summary Statistics, Raw pip data).
    """
    v_cols = ['open_violation_desc', 'high_violation_desc', 'low_violation_desc', 'close_violation_desc']
    info = pd.DataFrame(index=df.index)
    all_pips = []

    for col in v_cols:
        if col not in df.columns:
            continue
        
        # Extract numeric pips using regex
        def _extract_pips(val):
            if pd.isna(val): return np.nan
            match = re.search(r'(\d+\.?\d*)\s*pips', str(val))
            return abs(float(match.group(1))) if match else np.nan

        pip_values = df[col].apply(_extract_pips)
        info[col] = pip_values
        all_pips.extend(pip_values.dropna().tolist())

    if not all_pips:
        return pd.DataFrame(), info
    
    p_series = pd.Series(all_pips)
    bins = [0, 2, 5, 10, 20, 50, 100, 200, 500, 1000, float('inf')]
    labels = ['0-2', '2-5', '5-10', '10-20', '20-50', '50-100', '100-200', '200-500', '500-1000', '1000+']
    
    dist = pd.cut(p_series, bins=bins, labels=labels, right=False)
    counts = dist.value_counts().sort_index()
    pcts = (counts / len(all_pips) * 100).round(2)
    
    summary = pd.DataFrame({
        'viol_count': counts.values,
        'percentage': pcts.values
    }, index=labels)
    summary.index.name = 'pip_range'
    
    return summary, info

# ==============================================================================
# 4. REPORTING UTILITIES
# ==============================================================================

def create_report(df_dict: Dict[str, pd.DataFrame], filename: str) -> None:
    """
    Consolidates multiple validation dataframes into a readable text report.
    """
    with open(filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FOREX DATA QUALITY & SANITY REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        for title, df in df_dict.items():
            f.write(f"{title.upper()}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Shape: {len(df)} rows\n")
            
            if not df.empty:
                table = tabulate(df, headers='keys', tablefmt='simple', showindex=True, floatfmt=".4f")
                f.write(f"\n{table}\n")
            else:
                f.write("(No anomalies detected in this category)\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
