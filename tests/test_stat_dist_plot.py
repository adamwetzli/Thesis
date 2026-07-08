from utils import plot_multiple_financial_distributions
from numpy.random import default_rng
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

# Generate random data for 9 different statistics
rng = default_rng()

# Create 9 different distributions
data_dict = {
    'total_return': np.random.normal(0.15, 0.05, size=1000),           # 15% avg return, 5% std
    'sharpe': np.random.normal(1.4, 0.3, size=1000),                   # Sharpe ~1.4 ± 0.3
    'probabilistic_sharpe': np.random.beta(5, 2, size=1000),           # Beta distribution (0-1)
    'deflated_sharpe': np.random.beta(4, 2, size=1000),                # Beta distribution (0-1)
    'min_trl': np.random.exponential(0.1, size=1000),                  # Exponential (positive)
    'max_dd': -np.abs(np.random.normal(0.12, 0.03, size=1000)),        # Negative drawdown
    'cagr': np.random.normal(0.18, 0.04, size=1000),                   # 18% CAGR ± 4%
    'win_rate': np.random.beta(3, 3, size=1000),                       # Win rate 30-70%
    'profit_factor': np.random.normal(1.3, 0.2, size=1000),            # Profit factor ~1.3
    'n_trades': np.random.poisson(150, size=1000),               # ~150 trades
    'avg_capital_exposure': np.random.uniform(30, 60, size=1000),    # 30-60% exposure
    'avg_trade_size': np.random.uniform(30, 60, size=1000),          # $2500 ± $500
}

# Create custom bin edges for each statistic (optional)
bin_dict = {}
for stat_name, values in data_dict.items():
    # Create 30 bins between min and max for each distribution
    bin_dict[stat_name] = np.linspace(min(values), max(values), 31)

# Plot all 9 distributions
plot_multiple_financial_distributions(
    data_dict, 
    model_name='Test',
    bins_dict=bin_dict,  # Use custom bins (or set to None for auto-binning)
    show_stats=True,
    title="Test"
)


print(f"Generated {len(data_dict)} plots in a grid")