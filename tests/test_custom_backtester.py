from custom_backtester import CustomBacktester
from utils import plot_backtest_results
import numpy as np
import pandas as pd

# Set seeds for reproducibility
rng = np.random.default_rng()

def calculate_atr(df, window=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

size = 500

# 1. Artificial Signals (-1, 0, 1)
signals = pd.Series(rng.choice([-1, 0, 1], size=size))

# 2. Artificial M2 Confidence (probabilities between 0.5 and 0.95)
confidence = pd.Series(rng.uniform(0.5, 0.95, size=size))

# Load real OHLC prices (e.g. AUDUSD)
pair = "AUDUSD"
filepath = f"./data/csv_files/forex_combined/{pair}_combined_raw.csv"
full_data = pd.read_csv(filepath)
full_data.index = pd.to_datetime(full_data["date"])

ohlc = full_data[["open", "high", "low", "close"]].copy().iloc[:size]

# 3. Calculate ATR-based Distances
# TP = 3.0 * ATR, SL = 1.5 * ATR
atr = calculate_atr(ohlc)
tp_dist = atr * 3.0
sl_dist = atr * 1.5

# Initialize Backtester
print(f"\n--- Starting Professional ATR-Sized Backtest on {pair} ---")
bt = CustomBacktester(ohlc=ohlc, signals=signals, initial_cash=10_000.0)

# Run with Symmetric ATR Distances
# - Risk 2% of equity per trade
# - is_distance=True tells the backtester that sl/tp are price distances, not percentages
stats = bt.run(
    sl=sl_dist, 
    tp=tp_dist, 
    max_holding_periods=10,
    risk_pct=0.02,
    conformal_confidence=confidence, # Mocking confidence as 1 - p_value
    significance=0.1,
    kelly_fraction=0.5,
    min_qty=0,
    max_qty=100000.0,
    is_distance=True
)

# Output Analysis
print("\n[PERFORMANCE STATS]")
for k, v in stats.items():
    if k != 'portfolio_df':
        print(f"{k:15}: {v}")

# Verify Position Sizing Variation
if bt.trade_history:
    trades_df = pd.DataFrame(bt.trade_history)
    print("\n[POSITION SIZING SAMPLES]")
    # Calculate stop loss distance in pips for verification
    # For AUDUSD, 1 pip = 0.0001
    print(trades_df[['direction', 'size', 'net_pnl']].head(10))
    print(f"\nMean Trade Size: {trades_df['size'].abs().mean():.2f}")
    print(f"Max Trade Size:  {trades_df['size'].abs().max():.2f}")
    print(f"Min Trade Size:  {trades_df['size'].abs().min():.2f}")

plot_backtest_results(bt.ohlc, bt.trade_history, bt.equity_history, bt.initial_cash, stats=stats)
