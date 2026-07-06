"""
Forex Data Acquisition Module
=============================

This module provides functions to fetch historical forex data (OHLCV) from multiple sources,
including OpenBB (Polygon) and Dukascopy. It handles raw transaction data, bid/ask
spreads, and provides tools for merging and updating local datasets.

Core Functionality:
1. OpenBB Fetching: Hourly transaction data for various currency pairs.
2. Dukascopy Fetching: Hourly bid/ask OHLCV data from binary .bi5 files.
3. Data Merging: Consolidation of transaction and bid/ask data into master CSVs.

Institutional Standard:
- Rate-limit aware fetching (Polygon/OpenBB).
- Binary decoding for high-precision Dukascopy data.
- Automated duplicate removal and gap detection.
"""

import os
import time
import glob
import lzma
import struct
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Dict, List, Union, Optional
from openbb import obb

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

RAW_DIR = "./data/csv_files/forex_raw"
BID_ASK_DIR = "./data/csv_files/forex_bid_ask_raw"
COMBINED_DIR = "./data/csv_files/forex_combined"
BI5_CACHE_DIR = "./data/bi5_files"

# Blacklist for currencies not supported by one of the secondary sources
DUKASCOPY_BLACKLIST = {
    "USDINR": "no data on dukascopy",
    "USDKRW": "no data on dukascopy",
    "USDBRL": "no data on dukascopy",
    "USDRUB": "no data on dukascopy"
}

# ==============================================================================
# 1. MAIN ENTRY POINT
# ==============================================================================

def get_raw_forex_data(currencies: List[str], start_date: Union[str, datetime]) -> None:
    """
    Orchestrates the complete data acquisition pipeline.

    Args:
        currencies (List[str]): List of currency pairs to process (e.g., ['EURUSD', 'GBPUSD']).
        start_date (Union[str, datetime]): The earliest date to begin fetching from.
    """
    # Step 1: Fetch transaction data via OpenBB
    fetch_forex_from_openbb(currencies, start_date)

    # Step 2: Fetch bid/ask data from Dukascopy
    fetch_forex_from_dukascopy()

    # Step 3: Merge sources into unified datasets
    merge_forex_data()

# ==============================================================================
# 2. OPENBB (POLYGON) DATA ACQUISITION
# ==============================================================================

def fetch_forex_from_openbb(currencies: List[str], start_date: Union[str, datetime]) -> None:
    """
    Fetches hourly transaction OHLCV data from OpenBB/Polygon.

    Iterates through the provided currency list, filters against the blacklist,
    and manages the fetching process including directory creation and timing.

    Args:
        currencies (List[str]): Candidate currency pairs.
        start_date (Union[str, datetime]): Global start date for the fetch.
    """
    active_currencies = [c for c in currencies if c not in DUKASCOPY_BLACKLIST]
    os.makedirs(RAW_DIR, exist_ok=True)

    start_str, end_str = get_dates(start_date=start_date)
    start_time = datetime.now()

    print("\n" + "="*80)
    print("OPENBB FOREX DATA ACQUISITION")
    print(f"Started at: {start_time}")
    print(f"Window:     {start_str} to {end_str}")
    print("="*80)

    for i, currency in enumerate(active_currencies):
        print(f"\n[{i+1}/{len(active_currencies)}] Processing: {currency}")
        print("-" * 40)

        try:
            get_data(start_str, end_str, currency)
        except KeyboardInterrupt:
            print("\n!!! Process interrupted by user. Safely exiting...")
            break
        except Exception as e:
            print(f"!!! Error processing {currency}: {e}")
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print("OPENBB ACQUISITION COMPLETE")
    print(f"Finished at: {end_time}")
    print(f"Total time:  {str(duration).split('.')[0]}")
    print("="*80)

def get_dates(start_date: Union[str, datetime]) -> Tuple[str, str]:
    """
    Calculates the boundary dates for the data request.

    Ensures the end date is the most recent complete business day.

    Args:
        start_date (Union[str, datetime]): The requested start point.

    Returns:
        Tuple[str, str]: (start_date_string, end_date_string) in YYYY-MM-DD format.
    """
    if isinstance(start_date, str):
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    else:
        start_dt = start_date

    end_date = (datetime.now() - timedelta(days=1)).date()
    # Adjust to last Friday if yesterday was a weekend
    end_date = end_date - timedelta(days=max(0, end_date.weekday() - 4))
    
    return start_dt.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def get_data(start: str, end: str, currency: str) -> None:
    """
    Manages the incremental update of a single currency CSV file.

    Loads existing data if available, identifies the gap until 'end',
    and fetches missing periods in 15-second intervals to respect rate limits.

    Args:
        start (str): Fallback start date if no file exists.
        end (str): The target end date for the dataset.
        currency (str): The currency pair symbol.
    """
    file_path = f"{RAW_DIR}/{currency}_raw.csv"
    
    def _request_openbb(s_date: str, e_date: str, symbol: str) -> pd.DataFrame:
        """Internal helper to handle OpenBB API calls with retry logic."""
        attempt = 1
        while True:
            try:
                data = obb.currency.price.historical(
                    symbol=symbol,
                    start_date=s_date,
                    end_date=e_date,
                    provider="polygon",
                    interval="1h"
                )
                df = data.to_df()
                if not df.empty:
                    # Normalize index to timezone-naive UTC to avoid string-matching issues
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    return df
                
                print(f"   ... Empty response for {symbol}, retrying ({attempt})...")
                time.sleep(5 * attempt)
                attempt += 1
                if attempt > 3: return pd.DataFrame()
            except Exception as e:
                print(f"   ... API Error: {e}. Waiting to retry...")
                time.sleep(10)
                attempt += 1
                if attempt > 3: return pd.DataFrame()

    # Load or Initialize DataFrame
    try:
        df = pd.read_csv(file_path, index_col="date")
        df.index = pd.to_datetime(df.index).tz_localize(None)
    except FileNotFoundError:
        print(f"   ... No local data for {currency}. Starting initial fetch.")
        df = _request_openbb(start, end, currency)
        if df.empty:
            print(f"   ... Failed to fetch initial data for {currency}.")
            return

    target_dt = pd.to_datetime(end)
    print(f"   ... Current local horizon: {df.index[-1].strftime('%Y-%m-%d')}")
    
    # Incremental Update Loop
    while df.index[-1] < target_dt:
        time.sleep(15)  # Rate limit protection
        current_horizon = df.index[-1].strftime('%Y-%m-%d')
        new_df = _request_openbb(current_horizon, end, currency)
        
        if new_df.empty:
            print("   ... No new data available. Adjusting window.")
            new_end = (df.index[-1] + timedelta(days=30)).strftime('%Y-%m-%d')
            new_df = _request_openbb(current_horizon, min(new_end, end), currency)
            if new_df.empty: break
        else:
            print("   ... New data added.")

        df = pd.concat([df, new_df])
        
        # MANDATORY: Sort before dropping duplicates to ensure 'keep=last' is chronological
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]
        
        print(f"   ... Updated horizon: {df.index[-1].strftime('%Y-%m-%d')}")
    
    # Final Persistence
    df.to_csv(file_path)
    print(f"   ... Saved {currency} to {file_path}. Total rows: {len(df)}")

# ==============================================================================
# 3. DUKASCOPY (BID/ASK) DATA ACQUISITION
# ==============================================================================

def fetch_forex_from_dukascopy() -> None:
    """
    Iterates through locally available raw CSVs and fetches matching Bid/Ask data.

    Decodes binary .bi5 files from Dukascopy's servers for high-precision
    historical bid and ask pricing.
    """
    files = glob.glob(f"{RAW_DIR}/*.csv")
    start_time = datetime.now()

    print("\n" + "="*80)
    print("DUKASCOPY BID/ASK ACQUISITION")
    print(f"Started at: {start_time}")
    print("="*80)

    for i, file in enumerate(files):
        try:
            raw = pd.read_csv(file, index_col="date")
            raw.index = pd.to_datetime(raw.index).tz_localize(None)
            pair_name = os.path.basename(file).replace('_raw.csv', '')
            
            print(f"\n[{i+1}/{len(files)}] Processing Bid/Ask for: {pair_name}")
            print("-" * 40)

            start_date = raw.index[0]
            end_date = raw.index[-1]
            
            month_pairs = get_month_year_pairs_std(start_date, end_date)
            # Remove last month as it might be incomplete/unavailable on datafeed
            if month_pairs:
                del month_pairs[-1]
            
            get_bid_ask_ohlcv(pair_name, month_pairs)
        except Exception as e:
            print(f"!!! Error processing Dukascopy data for {file}: {e}")

    duration = datetime.now() - start_time
    print("\n" + "="*80)
    print("DUKASCOPY ACQUISITION COMPLETE")
    print(f"Total time: {str(duration).split('.')[0]}")
    print("="*80)

def get_bid_ask_ohlcv(symbol: str, dates: List[List[int]]) -> None:
    """
    Downloads and decodes binary bi5 data for a specific symbol and date range.

    Args:
        symbol (str): Currency pair (e.g., 'EURUSD').
        dates (List[List[int]]): List of [year, month] pairs to fetch.
    """
    bid_df_list = []
    ask_df_list = []

    symbol_cache_dir = f"{BI5_CACHE_DIR}/{symbol}"
    os.makedirs(symbol_cache_dir, exist_ok=True)
    os.makedirs(BID_ASK_DIR, exist_ok=True)

    for year, month in dates:
        for side in ["BID", "ASK"]:
            # Dukascopy uses 0-indexed months (00=Jan, 11=Dec)
            adj_month = str(month - 1).zfill(2)
            url = f"https://datafeed.dukascopy.com/datafeed/{symbol}/{year}/{adj_month}/{side}_candles_hour_1.bi5"
            cache_path = Path(f"{symbol_cache_dir}/{symbol}_{side}_{year}_{month}.bi5")

            def _download_bi5(target_url: str, save_path: Path):
                """Internal helper for downloading binary files."""
                try:
                    print(f"   ... Fetching {save_path.name}")
                    response = requests.get(target_url, timeout=30)
                    if response.status_code == 200:
                        with open(save_path, 'wb') as f:
                            f.write(response.content)
                    else:
                        print(f"   ... Warning: Received status {response.status_code} for {target_url}")
                except Exception as e:
                    print(f"   ... Download failed: {e}")

            # Check cache and download if needed
            if not cache_path.exists() or cache_path.stat().st_size == 0:
                _download_bi5(url, cache_path)
            
            if not cache_path.exists(): continue

            # Decode Binary Format
            try:
                # Reverted to user's original mapping: [Time, Open, Close, Low, High, Volume]
                df = bi5_to_df(str(cache_path), '>5If')
                
                # Scaling logic (Standard vs JPY)
                price_divider = 1000 if symbol == "USDJPY" else 100000

                # Column mapping from original script
                df.columns = ['seconds_from_start', 'open_raw', 'close_raw', 'low_raw', 'high_raw', f'{side}_volume_millions']
                df[f'{side}_open'] = df['open_raw'] / price_divider
                df[f'{side}_close'] = df['close_raw'] / price_divider
                df[f'{side}_low'] = df['low_raw'] / price_divider
                df[f'{side}_high'] = df['high_raw'] / price_divider

                base_date = pd.Timestamp(f'{year}-{month}-01', tz='UTC')
                df['date'] = base_date + pd.to_timedelta(df['seconds_from_start'], unit='s')
                
                df = df[['date', f'{side}_open', f'{side}_high', f'{side}_low', f'{side}_close', f'{side}_volume_millions']]
                
                if side == "BID": bid_df_list.append(df)
                else: ask_df_list.append(df)
            except Exception as e:
                print(f"   ... Error decoding {cache_path.name}: {e}")

    # Process and Save results with duplicate cleaning to prevent Cartesian joins
    for side, df_list in [("bid", bid_df_list), ("ask", ask_df_list)]:
        if df_list:
            final_df = pd.concat(df_list).set_index('date')
            final_df.index = pd.to_datetime(final_df.index).tz_localize(None)
            final_df = final_df.sort_index()
            # Remove duplicates that could occur at month boundaries in .bi5 files
            final_df = final_df[~final_df.index.duplicated(keep='last')]
            final_df.to_csv(f"{BID_ASK_DIR}/{symbol}_{side}_ohlcv_raw.csv")
            
    print(f"   ... Cleaned Bid/Ask datasets saved for {symbol}")

def bi5_to_df(filename: str, fmt: str) -> pd.DataFrame:
    """
    Decodes a Dukascopy .bi5 binary file (LZMA compressed).

    Args:
        filename (str): Path to the compressed binary file.
        fmt (str): Struct format string for unpacking.

    Returns:
        pd.DataFrame: Unpacked data.
    """
    chunk_size = struct.calcsize(fmt)
    data = []
    try:
        with lzma.open(filename) as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk: break
                data.append(struct.unpack(fmt, chunk))
    except Exception as e:
        raise ValueError(f"Failed to decompress or unpack {filename}: {e}")
    
    return pd.DataFrame(data)

def get_month_year_pairs_std(start_date: Union[str, datetime], end_date: Union[str, datetime]) -> List[List[int]]:
    """
    Generates a list of [Year, Month] pairs between two dates.

    Args:
        start_date: Start of the range.
        end_date: End of the range.

    Returns:
        List[List[int]]: E.g., [[2024, 1], [2024, 2], ...]
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    pairs = []
    curr_year, curr_month = start_date.year, start_date.month
    end_year, end_month = end_date.year, end_date.month
    
    while (curr_year < end_year) or (curr_year == end_year and curr_month <= end_month):
        pairs.append([curr_year, curr_month])
        curr_month += 1
        if curr_month > 12:
            curr_month = 1
            curr_year += 1
    return pairs

# ==============================================================================
# 4. DATA MERGER
# ==============================================================================

def merge_forex_data() -> None:
    """
    Combines primary transaction data with secondary bid/ask data into a master file.
    
    Ensures all source dataframes are clean and synchronized to naive UTC to
    prevent timezone-related join failures or duplicate multiplication.
    """
    raw_t, raw_ba = get_dicts()
    os.makedirs(COMBINED_DIR, exist_ok=True)

    print("\n" + "="*80)
    print("DATA MERGE OPERATION")
    print("="*80)

    for key, base_df in raw_t.items():
        try:
            # Safety Check: Deduplicate base before joining
            base_df = base_df.sort_index()
            base_df = base_df[~base_df.index.duplicated(keep='last')]
            
            for side in ["bid", "ask"]:
                side_key = f"{key}_{side}"
                if side_key in raw_ba:
                    side_df = raw_ba[side_key]
                    # Ensure join index matches base_df exactly (naive UTC)
                    base_df = base_df.join(side_df, how="left")
            
            # Post-join safety: ensure no rows were accidentally multiplied
            base_df = base_df[~base_df.index.duplicated(keep='last')]
            
            output_path = f"{COMBINED_DIR}/{key}_combined_raw.csv"
            base_df.to_csv(output_path)
            print(f"   ... Successfully merged {key} -> {output_path} (Rows: {len(base_df)})")
        except Exception as e:
            print(f"!!! Failed to merge {key}: {e}")

def get_dicts() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Loads all local CSV files into dictionaries for efficient merging.
    
    Normalizes all indices to naive UTC datetime objects.

    Returns:
        Tuple[Dict, Dict]: (Transaction Dictionary, Bid/Ask Dictionary)
    """
    files_t = glob.glob(f"{RAW_DIR}/*.csv")
    files_ba = glob.glob(f"{BID_ASK_DIR}/*.csv")
    
    raw_t = {}
    raw_ba = {}

    for file in files_t:
        df = pd.read_csv(file, index_col="date")
        df.index = pd.to_datetime(df.index).tz_localize(None)
        pair_name = os.path.basename(file).replace('_raw.csv', '')
        raw_t[pair_name] = df

    for file in files_ba:
        df = pd.read_csv(file, index_col="date")
        df.index = pd.to_datetime(df.index).tz_localize(None)
        pair_name = os.path.basename(file).replace('_ohlcv_raw.csv', '')
        raw_ba[pair_name] = df
    
    return raw_t, raw_ba
