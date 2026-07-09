import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from typing import List, Dict

from utils import plot_nested_wfv_dashboard


def create_synthetic_data_for_pair(pair_name: str, start_date: str, periods: int, 
                                   seed: int, price_base: float, volatility: float,
                                   trend_strength: float = 0.0001, 
                                   trade_frequency: int = 45,
                                   win_rate_bias: float = 0.5) -> Dict:
    """
    Creates synthetic trading data for a single currency pair.
    
    Parameters:
    - pair_name: Name of the currency pair (e.g., 'EURUSD')
    - start_date: Start date for the data
    - periods: Number of hourly periods
    - seed: Random seed for reproducibility
    - price_base: Starting price (e.g., 1.10 for EURUSD)
    - volatility: Price volatility (standard deviation of returns)
    - trend_strength: Strength of the trend (drift term)
    - trade_frequency: How often to generate trades (smaller = more trades)
    - win_rate_bias: Bias towards winning trades (0.5 = neutral, >0.5 = more winners)
    """
    np.random.seed(seed)
    
    # Create time index
    dates = pd.date_range(start=start_date, periods=periods, freq="h")
    
    # Generate price with trend and volatility
    returns = np.random.normal(trend_strength, volatility, size=len(dates))
    price = price_base * np.exp(np.cumsum(returns))
    
    ohlc = pd.DataFrame({
        "open": price,
        "high": price * (1 + np.abs(np.random.normal(0, 0.0005, size=len(dates)))),
        "low": price * (1 - np.abs(np.random.normal(0, 0.0005, size=len(dates)))),
        "close": price
    }, index=dates)
    
    # Generate trade history with varying performance
    trade_history = []
    trade_outcomes = []
    
    # Scale PnL based on price base (different for JPY pairs which have higher prices)
    if 'JPY' in pair_name:
        avg_pnl_scale = 20000
    elif 'XAU' in pair_name or 'GOLD' in pair_name:
        avg_pnl_scale = 50000
    else:
        avg_pnl_scale = 10000
    
    for i in range(10, len(dates)-10, trade_frequency):
        # Bias direction based on trend
        if trend_strength > 0:
            direction = "long" if np.random.rand() < (0.5 + trend_strength * 50) else "short"
        else:
            direction = "long" if np.random.rand() > 0.5 else "short"
        
        entry_idx = i
        hold_period = np.random.randint(15, 35)
        exit_idx = min(i + hold_period, len(dates) - 1)
        
        entry_price = ohlc["close"].iloc[entry_idx]
        exit_price = ohlc["close"].iloc[exit_idx]
        
        # Calculate PnL with some randomness
        raw_pnl = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
        
        # Bias the PnL to achieve desired win rate
        if np.random.rand() < win_rate_bias:
            # Make it a winning trade with some boost
            if raw_pnl < 0:
                raw_pnl = abs(raw_pnl) * 0.5
        else:
            # Make it a losing trade
            if raw_pnl > 0:
                raw_pnl = -abs(raw_pnl) * 0.8
        
        # Add noise
        noise = np.random.normal(0, abs(raw_pnl) * 0.1)
        pnl = raw_pnl + noise
        net_pnl = pnl * avg_pnl_scale
        
        # Track outcome for stats
        trade_outcomes.append(pnl > 0)
        
        trade_history.append({
            "direction": direction,
            "entry_time": dates[entry_idx],
            "exit_time": dates[exit_idx],
            "entry_price_raw": entry_price,
            "exit_price_raw": exit_price,
            "entry_price_slippage": entry_price,
            "net_pnl": net_pnl,
            "size": 1000
        })
    
    # Generate equity curve that follows trade performance
    initial_cash = 100000
    equity = [initial_cash]
    current_equity = initial_cash
    
    # Create equity changes based on trade outcomes
    for i in range(1, len(dates)):
        # Find trades that closed at this time
        trades_closed = [t for t in trade_history if t['exit_time'] == dates[i]]
        if trades_closed:
            for trade in trades_closed:
                current_equity += trade['net_pnl']
        
        # Add small random walk for realism
        current_equity *= (1 + np.random.normal(0, 0.0005))
        equity.append(current_equity)
    
    equity_df = pd.DataFrame({"equity": equity}, index=dates)
    
    # Calculate actual stats from the generated data
    total_return = (equity[-1] / initial_cash) - 1
    actual_win_rate = np.mean(trade_outcomes) if trade_outcomes else 0.5
    
    # Calculate Sharpe ratio
    equity_returns = equity_df['equity'].pct_change().dropna()
    if len(equity_returns) > 0 and equity_returns.std() > 0:
        sharpe = (equity_returns.mean() / equity_returns.std()) * np.sqrt(252 * 24)
        sharpe = min(max(sharpe, 0.5), 3.0)  # Clamp between 0.5 and 3.0
    else:
        sharpe = 1.0
    
    # Calculate max drawdown
    cumulative = equity_df['equity'] / initial_cash
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    stats = {
        "total_return": total_return,
        "sharpe": sharpe,
        "probabilistic_sharpe": min(sharpe * 0.7, 0.95),
        "deflated_sharpe": min(sharpe * 0.6, 0.85),
        "max_dd": max_dd,
        "cagr": total_return * (252 * 24 / periods) if total_return > -1 else -0.5,
        "win_rate": actual_win_rate,
        "profit_factor": max(1.2, np.random.uniform(1.3, 2.5)),
        "n_trades": len(trade_history),
        "avg_capital_exposure": np.random.uniform(15, 35),
        "avg_trade_size": np.random.uniform(3, 8)
    }
    
    return {
        "ohlc": ohlc,
        "trade_history": trade_history,
        "equity_history": equity_df,
        "stats": stats,
        "initial_cash": initial_cash
    }


def generate_pair_configurations(n_pairs: int = 10) -> Dict:
    """
    Generates configurations for multiple currency pairs dynamically.
    
    Parameters:
    - n_pairs: Number of currency pairs to generate
    
    Returns:
    - Dictionary mapping pair names to their configurations
    """
    
    # Base currencies and quote currencies for realistic pair generation
    base_currencies = ['EUR', 'GBP', 'AUD', 'NZD', 'CAD', 'CHF', 'USD', 'JPY']
    quote_currencies = ['USD', 'JPY', 'GBP', 'CHF', 'AUD', 'CAD', 'EUR']
    
    # Avoid duplicates and create unique pairs
    pairs_config = {}
    used_pairs = set()
    
    # Predefined major pairs with realistic values
    major_pairs = [
        ('EURUSD', 1.10, 0.0012, 0.0001, 0.55),
        ('GBPUSD', 1.27, 0.0011, 0.00008, 0.53),
        ('USDJPY', 150.50, 0.0009, 0.00005, 0.52),
        ('AUDUSD', 0.66, 0.0013, 0.00007, 0.51),
        ('USDCAD', 1.35, 0.0010, 0.00006, 0.50),
        ('NZDUSD', 0.61, 0.0012, 0.00007, 0.51),
        ('USDCHF', 0.89, 0.0009, 0.00004, 0.50),
        ('EURGBP', 0.87, 0.0008, 0.00003, 0.52),
        ('EURJPY', 165.50, 0.0010, 0.00007, 0.54),
        ('GBPJPY', 190.20, 0.0011, 0.00009, 0.53),
        ('AUDJPY', 99.30, 0.0012, 0.00008, 0.52),
        ('CADJPY', 111.40, 0.0011, 0.00007, 0.51),
        ('CHFJPY', 169.80, 0.0009, 0.00005, 0.50),
        ('EURCAD', 1.49, 0.0010, 0.00006, 0.51),
        ('GBPAUD', 1.92, 0.0012, 0.00010, 0.53),
        ('XAUUSD', 2350.00, 0.0015, 0.00015, 0.48),  # Gold
        ('XAGUSD', 28.50, 0.0016, 0.00012, 0.47),   # Silver
        ('BTCUSD', 65000, 0.0020, 0.00020, 0.45),   # Bitcoin
        ('ETHUSD', 3500, 0.0018, 0.00018, 0.46),    # Ethereum
        ('USDSGD', 1.35, 0.0008, 0.00004, 0.50),
        ('USDHKD', 7.82, 0.0003, 0.00001, 0.51),
        ('USDMXN', 16.80, 0.0014, 0.00012, 0.49),
        ('USDZAR', 18.50, 0.0015, 0.00013, 0.48),
        ('USDTRY', 32.50, 0.0019, 0.00018, 0.47),
        ('USDPLN', 4.02, 0.0011, 0.00009, 0.50),
        ('USDDKK', 6.89, 0.0007, 0.00003, 0.51),
        ('USDSEK', 10.65, 0.0010, 0.00008, 0.50),
        ('USDNOK', 10.80, 0.0011, 0.00009, 0.49),
    ]
    
    # Use major pairs first, then generate additional synthetic ones if needed
    for i in range(min(n_pairs, len(major_pairs))):
        name, price, vol, trend, win_bias = major_pairs[i]
        pairs_config[name] = {
            'price_base': price,
            'volatility': vol,
            'trend': trend,
            'seed_offset': i * 100,
            'win_rate_bias': win_bias,
            'trade_frequency': np.random.randint(35, 55)
        }
    
    # If more pairs needed, generate synthetic ones
    if n_pairs > len(major_pairs):
        synthetic_idx = len(major_pairs)
        for i in range(n_pairs - len(major_pairs)):
            # Generate random but realistic pair
            base = np.random.choice(base_currencies)
            quote = np.random.choice([q for q in quote_currencies if q != base])
            pair_name = f"{base}{quote}"
            
            # Avoid duplicates
            while pair_name in pairs_config:
                base = np.random.choice(base_currencies)
                quote = np.random.choice([q for q in quote_currencies if q != base])
                pair_name = f"{base}{quote}"
            
            # Generate realistic parameters
            if 'JPY' in quote:
                price_base = np.random.uniform(100, 200)
                volatility = np.random.uniform(0.0008, 0.0012)
            elif base == 'EUR':
                price_base = np.random.uniform(0.85, 1.20)
                volatility = np.random.uniform(0.0007, 0.0011)
            elif base == 'GBP':
                price_base = np.random.uniform(1.20, 1.35)
                volatility = np.random.uniform(0.0008, 0.0012)
            else:
                price_base = np.random.uniform(0.60, 1.50)
                volatility = np.random.uniform(0.0007, 0.0013)
            
            pairs_config[pair_name] = {
                'price_base': price_base,
                'volatility': volatility,
                'trend': np.random.uniform(0.00003, 0.00012),
                'seed_offset': synthetic_idx * 100 + i * 10,
                'win_rate_bias': np.random.uniform(0.47, 0.58),
                'trade_frequency': np.random.randint(35, 65)
            }
    
    return pairs_config


def create_multi_pair_synthetic_data(n_folds: int = 2, n_pairs: int = 10) -> List[Dict[str, Dict]]:
    """
    Creates synthetic data for multiple currency pairs and multiple folds.
    
    Parameters:
    - n_folds: Number of folds (columns in dashboard)
    - n_pairs: Number of currency pairs (rows in dashboard)
    
    Returns:
    - List of fold dictionaries, each containing pair data
    """
    
    print(f"\n[INFO] Generating synthetic data for {n_pairs} currency pairs and {n_folds} folds...")
    
    # Generate pair configurations dynamically
    pairs_config = generate_pair_configurations(n_pairs)
    selected_pairs = list(pairs_config.keys())
    
    print(f"[INFO] Generated pairs: {', '.join(selected_pairs[:5])}{'...' if n_pairs > 5 else ''}")
    
    all_folds = []
    
    for fold_idx in range(n_folds):
        print(f"[INFO] Generating fold {fold_idx + 1}/{n_folds}...")
        fold_data = {}
        
        for pair_idx, pair_name in enumerate(selected_pairs):
            config = pairs_config[pair_name]
            
            # Different seed for each fold and pair to create variation
            seed = (fold_idx * 10000) + (pair_idx * 100) + config['seed_offset']
            
            # Different start dates for each fold
            if fold_idx == 0:
                start_date = "2024-01-01"
            elif fold_idx == 1:
                start_date = "2024-03-01"
            else:
                start_date = f"2024-{fold_idx * 2:02d}-01"
            
            # Shorter periods for testing with many pairs (to keep file size reasonable)
            periods = 300 if n_pairs > 5 else 500
            
            # Create data with pair-specific characteristics
            pair_data = create_synthetic_data_for_pair(
                pair_name=pair_name,
                start_date=start_date,
                periods=periods,
                seed=seed,
                price_base=config['price_base'],
                volatility=config['volatility'],
                trend_strength=config['trend'],
                trade_frequency=config.get('trade_frequency', 45),
                win_rate_bias=config.get('win_rate_bias', 0.5)
            )
            
            fold_data[pair_name] = pair_data
        
        all_folds.append(fold_data)
    
    print(f"[INFO] Data generation complete!")
    return all_folds


def test_dashboard_visuals():
    """
    Test the dashboard with multiple pairs and folds.
    """
    print("\n" + "="*80)
    print("TESTING WFV DASHBOARD VISUALS WITH MULTI-PAIR DATA")
    print("="*80)
    
    # Test Case 1: Single fold with 3 pairs (Quick test)
    single_fold_data = create_multi_pair_synthetic_data(n_folds=1, n_pairs=10)
    plot_nested_wfv_dashboard(
        model_name="Synthetic_Prod_10Pairs",
        all_fold_results=single_fold_data,
        title_pref="a_production"
    )
    
    # Test Case 2: Two folds with 5 pairs (Medium test)
    multi_fold_data = create_multi_pair_synthetic_data(n_folds=2, n_pairs=10)
    plot_nested_wfv_dashboard(
        model_name="Synthetic_Tourney_10Pairs",
        all_fold_results=multi_fold_data,
        title_pref="a_tournament"
    )

if __name__ == "__main__":
    test_dashboard_visuals()