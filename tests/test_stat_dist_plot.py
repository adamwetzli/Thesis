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
    'Sharpe Ratio': rng.normal(loc=0.5, scale=0.3, size=1000),
    'Win Rate': rng.beta(a=2, b=2, size=1000),  # Beta distribution for rates
    'Sortino Ratio': rng.normal(loc=0.4, scale=0.4, size=1000),
    'Max Drawdown': rng.uniform(low=-0.5, high=-0.05, size=1000),
    'Calmar Ratio': rng.exponential(scale=0.5, size=1000),
    'Volatility': rng.gamma(shape=2, scale=0.1, size=1000),
    'Alpha': rng.normal(loc=0.02, scale=0.05, size=1000),
    'Beta': rng.normal(loc=1.0, scale=0.2, size=1000),
    'Information Ratio': rng.normal(loc=0.3, scale=0.35, size=1000)
}

# Create custom bin edges for each statistic (optional)
bin_dict = {}
for stat_name, values in data_dict.items():
    # Create 30 bins between min and max for each distribution
    bin_dict[stat_name] = np.linspace(min(values), max(values), 31)

# Plot all 9 distributions
plot_multiple_financial_distributions(
    data_dict, 
    model_name='RandomForest',
    bins_dict=bin_dict,  # Use custom bins (or set to None for auto-binning)
    show_stats=True
)


print(f"Generated {len(data_dict)} plots in a grid")