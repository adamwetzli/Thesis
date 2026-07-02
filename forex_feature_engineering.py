"""
Forex Feature Engineering Module
================================

This module provides a comprehensive suite of technical indicators, statistical features,
and time-based session data for forex trading models. It implements institutional
naming conventions (num_, bin_, cyc_, cat_) to facilitate automated preprocessing.

Feature Categories:
1. Momentum Indicators (RSI, MACD, TSI, STC, etc.)
2. Overlap Indicators (VWMA)
3. Trend Indicators (ATR, ADX, Aroon)
4. Volatility Indicators (RVI)
5. Volume Indicators (CMF, Accumulation/Distribution)
6. Microstructural Features (Order Flow Imbalance, Spread Analysis)
7. Time-Session Features (London, NY, Asia session flags and cyclical encoding)

Institutional Standard:
- Fully vectorized computations for performance.
- Automated prefixing for downstream pipeline scaling/encoding.
- Protection against data loss in cross-pair correlation features.
"""

import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional, Dict, Union

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

CLEANED_DIR = "./csv_files/forex_cleaned_and_validated"
FEATURES_DIR = "./csv_files/forex_features_raw"

# Standard pip multipliers for specific currency pairs
PIP_MULTIPLIERS = {
    "USDJPY": 100,
    "EURJPY": 100,
    "GBPJPY": 100,
}
DEFAULT_PIP_MULT = 10000

# Institutional Classification for Global Model Context
PAIR_GROUPS = {
    "EURUSD": "Major", "GBPUSD": "Major", "USDJPY": "Major",
    "AUDUSD": "Major", "USDCAD": "Major", "USDCHF": "Major", "NZDUSD": "Major",
    "USDTRY": "EM", "USDRUB": "EM", "USDINR": "EM", "USDKRW": "EM",
    "USDBRL": "EM", "USDZAR": "EM", "USDMXN": "EM", "USDCNH": "EM"
}

# ==============================================================================
# 1. MAIN FEATURE PIPELINE
# ==============================================================================

def engineer_forex_features() -> None:
    """
    Orchestrates the feature engineering process for all available currency pairs.
    
    Loads cleaned data, applies a sequence of feature generators, and persists
    the resulting datasets for the preprocessing stage.
    """
    os.makedirs(FEATURES_DIR, exist_ok=True)
    files = glob.glob(f"{CLEANED_DIR}/*.csv")

    # Load all pairs to allow for potential cross-pair features
    pairs = {}
    for file in files:
        symbol = os.path.basename(file).replace("_clean.csv", "")
        df = pd.read_csv(file, index_col="date")
        df.index = pd.to_datetime(df.index)
        pairs[symbol] = df

    start_time = datetime.now()
    print("\n" + "="*80)
    print("FOREX FEATURE ENGINEERING PIPELINE")
    print(f"Started at: {start_time}")
    print("="*80)

    for i, (symbol, df_raw) in enumerate(pairs.items(), 1):
        print(f"\n[{i}/{len(pairs)}] Engineering features for: {symbol}")
        print("-" * 40)

        pip_multiplier = PIP_MULTIPLIERS.get(symbol, DEFAULT_PIP_MULT)
        
        # Start with a copy to prevent modifying the loaded 'pairs' dictionary
        df = df_raw.copy()
        
        # Step 0: Contextual Metadata (Pair Personality)
        df['cat_pair_group'] = PAIR_GROUPS.get(symbol, "Other")

        # Step 1: Temporal Context
        df = calculate_time_features(df, inplace=True)
        
        # Step 2: Technical Indicators (Prof. Morini Suite)
        df = calculate_momentum_indicators(df, inplace=True)
        df = calculate_volume_indicators(df, inplace=True)
        df = calculate_overlap_indicators(df, inplace=True)
        df = calculate_trend_indicators(df, inplace=True)
        df = calculate_volatility_indicators(df, inplace=True)
        df = calculate_more_indicators(df, inplace=True)

        # Step 3: Microstructural & Price Features
        df = calculate_price_features(df, pip_multiplier=pip_multiplier, inplace=True)
        
        # Step 4: Interaction Features
        df = calculate_more_interaction_features(df, inplace=True)

        # Save result
        output_path = f"{FEATURES_DIR}/{symbol}_features_raw.csv"
        df.to_csv(output_path)
        print(f"   ... Saved {len(df.columns)} features to {output_path}")

    duration = datetime.now() - start_time
    print("\n" + "="*80)
    print("FEATURE ENGINEERING COMPLETE")
    print(f"Total time: {str(duration).split('.')[0]}")
    print("="*80)

# ==============================================================================
# 2. TEMPORAL & CROSS-PAIR FEATURES
# ==============================================================================

def calculate_rolling_correlations(pairs: Dict[str, pd.DataFrame], 
                                   col: str = "tx_vwap", 
                                   window: int = 20) -> pd.DataFrame:
    """
    Calculates rolling correlations between all pairs for a specific column.
    """
    if not pairs:
        return pd.DataFrame()

    master_index = next(iter(pairs.values())).index
    data_matrix = pd.DataFrame(index=master_index)

    for symbol, df in pairs.items():
        if col in df.columns:
            data_matrix[symbol] = df[col]

    symbols = list(data_matrix.columns)
    corr_dict = {}

    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            s1, s2 = symbols[i], symbols[j]
            feat_name = f"pass_corr_{s1}_{s2}_w{window}"
            corr_dict[feat_name] = data_matrix[s1].rolling(window, min_periods=window//2).corr(data_matrix[s2])

    return pd.DataFrame(corr_dict, index=master_index)

def calculate_time_features(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    """
    Generates time-based features, session flags, and cyclical encodings.
    """
    if not inplace:
        df = df.copy()
    
    # 1. Basic Decomposition
    df['time_hour'] = df.index.hour
    df['time_day_of_week'] = df.index.dayofweek
    df['time_day_of_month'] = df.index.day
    df['time_week_of_year'] = df.index.isocalendar().week.astype(int)
    df['time_month'] = df.index.month
    df['bin_is_weekend'] = df['time_day_of_week'].isin([5, 6]).astype(int)
    
    # 2. Institutional Session Flags (UTC)
    df['bin_is_asia_session'] = ((df['time_hour'] >= 22) | (df['time_hour'] < 6)).astype(int)
    df['bin_is_london_session'] = ((df['time_hour'] >= 7) & (df['time_hour'] < 16)).astype(int)
    df['bin_is_ny_session'] = ((df['time_hour'] >= 12) & (df['time_hour'] < 21)).astype(int)
    
    # Session Overlaps
    df['bin_is_london_ny_overlap'] = ((df['time_hour'] >= 12) & (df['time_hour'] < 16)).astype(int)
    df['bin_is_asia_london_overlap'] = ((df['time_hour'] >= 7) & (df['time_hour'] < 8)).astype(int)
    
    # Specific Event Hours
    df['bin_is_london_open_hour'] = (df.index.hour == 7).astype(int)
    df['bin_is_ny_open_hour'] = (df.index.hour == 12).astype(int)
    df['bin_is_london_fix_hour'] = (df.index.hour == 16).astype(int)
    
    # 3. Session Transitions (Ordinal)
    df['ord_session_transition'] = 0
    df.loc[df['time_hour'] == 6, 'ord_session_transition'] = 1   # Asia -> London
    df.loc[df['time_hour'] == 12, 'ord_session_transition'] = 2  # London -> NY
    df.loc[df['time_hour'] == 21, 'ord_session_transition'] = 3  # NY -> Asia
    
    # 4. Categorical Time of Day
    df['cat_time_of_day'] = pd.cut(df['time_hour'], 
                                   bins=[0, 4, 8, 12, 16, 20, 24], 
                                   labels=['Late Night', 'Early Morning', 'Morning', 
                                           'Afternoon', 'Evening', 'Night'],
                                   include_lowest=True)
    
    # 5. Cyclical Encoding (Sin/Cos)
    df['cyc_hour_sin'] = np.sin(2 * np.pi * df['time_hour'] / 24)
    df['cyc_hour_cos'] = np.cos(2 * np.pi * df['time_hour'] / 24)
    df['cyc_day_sin'] = np.sin(2 * np.pi * df['time_day_of_week'] / 7)
    df['cyc_day_cos'] = np.cos(2 * np.pi * df['time_day_of_week'] / 7)
    df['cyc_week_sin'] = np.sin(2 * np.pi * (df['time_week_of_year'] - 1) / 52)
    df['cyc_week_cos'] = np.cos(2 * np.pi * (df['time_week_of_year'] - 1) / 52)
    df['cyc_month_sin'] = np.sin(2 * np.pi * (df['time_month'] - 1) / 12)
    df['cyc_month_cos'] = np.cos(2 * np.pi * (df['time_month'] - 1) / 12)

    # 6. Calendar Event Flags
    df['bin_is_end_of_month'] = (df.index.is_month_end).astype(int)
    df['bin_is_end_of_quarter'] = (df.index.is_quarter_end).astype(int)
    df['bin_is_start_of_month'] = (df.index.is_month_start).astype(int)

    return df

# ==============================================================================
# 3. TECHNICAL INDICATORS (MOMENTUM, VOLUME, TREND)
# ==============================================================================

def calculate_momentum_indicators(df: pd.DataFrame, 
                                  macd_short: int = 12, 
                                  macd_long: int = 26, 
                                  rsi_period: int = 14, 
                                  tsi_period: int = 25, 
                                  stc_period: int = 10, 
                                  stc_smooth: int = 5, 
                                  will_period: int = 14, 
                                  cfo_period: int = 14, 
                                  inplace: bool = True) -> pd.DataFrame:
    """
    Computes standard momentum indicators (RSI, MACD, TSI, STC, Williams %R, CFO).
    """
    if not inplace:
        df = df.copy()

    # MACD
    ema_short = df['tx_close'].ewm(span=macd_short, adjust=False).mean()
    ema_long = df['tx_close'].ewm(span=macd_long, adjust=False).mean()
    df['num_MACD'] = ema_short - ema_long
     
    # RSI
    delta = df['tx_close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = gain / (loss + 1e-9)
    df['num_RSI'] = 100 - (100 / (1 + rs))
    
    # True Strength Index (TSI)
    diff = df['tx_close'].diff()
    df['num_TSI'] = 100 * (diff.ewm(span=tsi_period).mean() / (diff.abs().ewm(span=tsi_period).mean() + 1e-9))
    df['num_SLOPE_TSI'] = df['num_TSI'].diff()
    
    # Relative Vigor Index (RVGI)
    rvgi_num = ((df['tx_close'] - df['tx_close'].shift(1)) +
                2 * (df['tx_close'].shift(1) - df['tx_close'].shift(2)) +
                (df['tx_close'].shift(2) - df['tx_close'].shift(3)))
    rvgi_den = ((df['tx_high'] - df['tx_low']) +
                2 * (df['tx_high'].shift(1) - df['tx_low'].shift(1)) +
                (df['tx_high'].shift(2) - df['tx_low'].shift(2)))
    df['num_RVGI'] = rvgi_num / (rvgi_den + 1e-9)
    
    # Schaff Trend Cycle (STC)
    stc_macd = df['tx_close'].ewm(span=macd_short).mean() - df['tx_close'].ewm(span=macd_long).mean()
    stc_min = stc_macd.rolling(stc_period).min()
    stc_max = stc_macd.rolling(stc_period).max()
    stc_k = 100 * ((stc_macd - stc_min) / (stc_max - stc_min + 1e-9))
    df['num_STC'] = stc_k.ewm(span=stc_smooth).mean()
    df['num_SLOPE_STC'] = df['num_STC'].diff()
    
    # Williams %R
    will_h = df['tx_high'].rolling(will_period).max()
    will_l = df['tx_low'].rolling(will_period).min()
    df['num_WILLIAMS_%R'] = ((will_h - df['tx_close']) / (will_h - will_l + 1e-9)) * -100
    
    # CFO
    cfo_rolling_mean = df['tx_close'].rolling(cfo_period).mean()
    df['num_CFO'] = (df['tx_close'] - cfo_rolling_mean) / (df['tx_close'] + 1e-9) * 100
    
    return df

def calculate_overlap_indicators(df: pd.DataFrame, 
                                 vmwa_period: int = 14, 
                                 inplace: bool = True) -> pd.DataFrame:
    """Computes Volume Weighted Moving Average (VWMA)."""
    if not inplace:
        df = df.copy()

    df['num_VWMA'] = (df['tx_close'] * df['tx_volume']).rolling(vmwa_period).sum() / \
                     (df['tx_volume'].rolling(vmwa_period).sum() + 1e-9)
    df['num_SLOPE_VWMA'] = df['num_VWMA'].diff() 
    
    return df

def calculate_trend_indicators(df: pd.DataFrame, 
                               atr_period: int = 14, 
                               aroon_period: int = 25, 
                               inplace: bool = True) -> pd.DataFrame:
    """Computes ATR, ADX proxy, and Aroon."""
    if not inplace:
        df = df.copy()

    df['num_TR'] = np.maximum(df['tx_high'] - df['tx_low'],
                              np.maximum(np.abs(df['tx_high'] - df['tx_close'].shift(1)),
                                         np.abs(df['tx_low'] - df['tx_close'].shift(1))))
    df['num_atr'] = df['num_TR'].rolling(atr_period).mean()
    df['num_ADX'] = (np.abs(df['tx_high'] - df['tx_low']).ewm(span=atr_period).mean()) / (df['num_atr'] + 1e-9)
    
    df['AROON_UP'] = df['tx_high'].rolling(aroon_period).apply(lambda x: (aroon_period - np.argmax(x[::-1])) / aroon_period * 100, raw=True)
    df['AROON_DOWN'] = df['tx_low'].rolling(aroon_period).apply(lambda x: (aroon_period - np.argmin(x[::-1])) / aroon_period * 100, raw=True)
    df['num_AROON'] = df['AROON_UP'] - df['AROON_DOWN']
    df.drop(columns=['AROON_UP', 'AROON_DOWN'], inplace=True)
    
    return df

def calculate_volatility_indicators(df: pd.DataFrame, 
                                    rvi_period: int = 14, 
                                    inplace: bool = True) -> pd.DataFrame:
    """Computes Relative Volatility Index (RVI)."""
    if not inplace:
        df = df.copy()

    up_moves = df['tx_close'].diff().apply(lambda x: x if x > 0 else 0)
    all_moves = df['tx_close'].diff().abs()
    df['num_RVI'] = (up_moves.rolling(rvi_period).std() / (all_moves.rolling(rvi_period).std() + 1e-9)) * 100
    
    return df

def calculate_volume_indicators(df: pd.DataFrame, 
                                cmf_period: int = 20, 
                                inplace: bool = True) -> pd.DataFrame:
    """Computes Chaikin Money Flow (CMF) and Accumulation/Distribution."""
    if not inplace:
        df = df.copy()

    # Money Flow Multiplier
    mf_mult = ((df['tx_close'] - df['tx_low']) - (df['tx_high'] - df['tx_close'])) / \
              (df['tx_high'] - df['tx_low'] + 1e-9)
    
    # A/D Line
    df['num_A/D'] = (mf_mult * df['tx_volume']).cumsum()
    # Normalize A/D to 0-1 range
    ad_min = df['num_A/D'].min()
    ad_max = df['num_A/D'].max()
    df['num_A/D'] = (df['num_A/D'] - ad_min) / (ad_max - ad_min + 1e-9)
    df['num_SLOPE_A/D'] = df['num_A/D'].diff()

    # CMF
    mf_volume = mf_mult * df['tx_volume']
    df['num_CMF'] = mf_volume.rolling(cmf_period).sum() / (df['tx_volume'].rolling(cmf_period).sum() + 1e-9)
    
    return df

def calculate_more_indicators(df: pd.DataFrame, 
                              dc_period: int = 20, 
                              inplace: bool = True) -> pd.DataFrame:
    """Computes Donchian Channel distance features."""
    if not inplace:
        df = df.copy()

    dc_u = df['tx_high'].rolling(dc_period).max()
    dc_l = df['tx_low'].rolling(dc_period).min()
    df['num_PRICE_FROM_DONCHIAN'] = (df['tx_close'] - dc_l) / (dc_u - dc_l + 1e-9)
    df['num_SLOPE_PRICE_FROM_DONCHIAN'] = df['num_PRICE_FROM_DONCHIAN'].diff()
    
    return df

# ==============================================================================
# 4. MICROSTRUCTURAL & INTERACTION FEATURES
# ==============================================================================

def calculate_price_features(df: pd.DataFrame, 
                             pip_multiplier: float, 
                             inplace: bool = True) -> pd.DataFrame:
    """Computes microstructural features including Bid/Ask spreads and OFI."""
    if not inplace:
        df = df.copy()
        
    # Mid-Price Context
    df['num_mid_close'] = (df['bid_close'] + df['ask_close']) / 2
    df['num_momentum_5'] = df['num_mid_close'].pct_change(5)
    df['num_momentum_20'] = df['num_mid_close'].pct_change(20)
    df['num_momentum_acceleration'] = df['num_momentum_5'].diff()

    # High/Low Breakouts
    df['num_mid_high'] = (df['bid_high'] + df['ask_high']) / 2
    df['num_mid_low'] = (df['bid_low'] + df['ask_low']) / 2
    df['bin_is_breaking_high'] = (df['num_mid_high'] > df['num_mid_high'].rolling(20).max().shift(1)).astype(int)
    df['bin_is_breaking_low'] = (df['num_mid_low'] < df['num_mid_low'].rolling(20).min().shift(1)).astype(int)

    # Volatility and Return Profiles
    df['num_mid_returns'] = df['num_mid_close'].pct_change()
    df['num_volatility_10'] = df['num_mid_returns'].rolling(10).std()
    df['num_volatility_50'] = df['num_mid_returns'].rolling(50).std()

    # Spread Analysis
    df['num_spread_abs'] = df['ask_close'] - df['bid_close']
    df['num_spread_pips'] = df['num_spread_abs'] * pip_multiplier
    df['num_normalized_spread'] = df['num_spread_pips'] / (df['num_mid_returns'].abs().rolling(20).std() * pip_multiplier + 1e-9)
    df['num_spread_volatility_ratio'] = df['num_spread_pips'] / (df['num_volatility_10'] * 10000 + 1e-9)
    df['num_spread_momentum'] = df['num_spread_pips'].diff()
    df['num_spread_volatility_10'] = df['num_spread_pips'].rolling(10).std()

    # Order Flow Imbalance (OFI)
    df['num_order_flow_imbalance'] = (df['bid_volume_mil'] - df['ask_volume_mil']) / \
                                    (df['bid_volume_mil'] + df['ask_volume_mil'] + 1e-7)
    df['num_ofi_rolling_mean_10'] = df['num_order_flow_imbalance'].rolling(10).mean()
    
    # Efficient Price (Information Content)
    df['num_efficient_price'] = (df['bid_close'] * df['ask_volume_mil'] + 
                                df['ask_close'] * df['bid_volume_mil']) / \
                                (df['bid_volume_mil'] + df['ask_volume_mil'] + 1e-7)
    df['num_efficient_price_deviation'] = df['num_efficient_price'] - df['num_mid_close']

    # Momentum Gaps
    df['num_bid_momentum_3'] = df['bid_close'].pct_change(3)
    df['num_ask_momentum_3'] = df['ask_close'].pct_change(3)
    df['num_bid_ask_momentum_gap'] = df['num_bid_momentum_3'] - df['num_ask_momentum_3']

    # Liquidity Dynamics
    df['num_volume_price_confirmation'] = np.sign(df['num_mid_returns']) * df['num_order_flow_imbalance']
    df['num_trade_position_in_spread'] = (df['tx_close'] - df['bid_close']) / (df['num_spread_abs'] + 1e-9)
    df['num_volume_total'] = df['bid_volume_mil'] + df['ask_volume_mil']
    df['num_volume_weighted_return'] = df['num_mid_returns'] * df['num_volume_total']
    df['num_price_impact'] = abs(df['num_mid_returns']) / (df['num_volume_total'] + 1e-9)
    df['num_quote_resilience'] = (df['ask_high'] - df['ask_low']) - (df['bid_high'] - df['bid_low'])

    # Session-Specific Trends
    df['num_london_session_trend'] = df['num_mid_returns'].where(df['bin_is_london_session'] == 1, 0).rolling(4).mean()
    df['num_ny_session_trend'] = df['num_mid_returns'].where(df['bin_is_ny_session'] == 1, 0).rolling(4).mean()
    df['num_diff_session_trend'] = df['num_ny_session_trend'] - df['num_london_session_trend']

    # Relative Channel Positioning
    df['num_distance_to_recent_high'] = (df['num_mid_close'] - df['num_mid_high'].rolling(12).max()) / (df['num_mid_close'] + 1e-9)
    df['num_distance_to_recent_low'] = (df['num_mid_close'] - df['num_mid_low'].rolling(12).min()) / (df['num_mid_close'] + 1e-9)
    
    return df

def calculate_more_interaction_features(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    """Computes interaction features between time sessions and market dynamics."""
    if not inplace:
        df = df.copy()
    
    # Session Volatility
    df['num_london_volatility'] = df['num_volatility_10'] * df['bin_is_london_session']
    df['num_ny_volatility'] = df['num_volatility_10'] * df['bin_is_ny_session']

    # Liquidity-Event Interactions
    df['num_ofi_during_london_fix'] = df['num_order_flow_imbalance'] * df['bin_is_london_fix_hour']
    df['num_price_impact_during_asia'] = df['num_price_impact'] * df['bin_is_asia_session']
    df['num_spread_during_overlap'] = df['num_spread_pips'] * df['bin_is_london_ny_overlap']

    return df
