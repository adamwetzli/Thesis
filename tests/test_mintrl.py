import numpy as np
import pandas as pd
from scipy.stats import norm

def calculate_mintrl(equity_series, benchmark_sr=0.0, alpha=0.05, periods_per_year=252*24):
    """
    Calculates the Minimum Track Record Length (MinTRL) in terms of number of observations.
    
    Formula from Marcos Lopez de Prado (2018):
    MinTRL = 1 + [1 - skew*SR + ((kurt-1)/4)*SR^2] * (Z_alpha / (SR - SR_bench))^2
    
    Note: SR and SR_bench used in the square brackets must be the 
    NON-ANNUALIZED (per-period) values.
    """
    if len(equity_series) < 4:
        return np.inf

    # 1. Calculate returns
    returns = equity_series.pct_change().dropna()
    if returns.std() == 0:
        return np.inf

    # 2. Calculate Moments (Non-annualized)
    n_obs = len(returns)
    sr_obs = returns.mean() / returns.std()
    sr_bench = benchmark_sr / np.sqrt(periods_per_year)
    
    # We need the Fisher Kurtosis (excess) + 3 to get Pearson Kurtosis
    skew = returns.skew()
    kurt = returns.kurtosis() + 3 
    
    # 3. Target Z-score (e.g., 1.645 for 95% confidence)
    z_alpha = norm.ppf(1 - alpha)
    
    # 4. Calculate MinTRL
    # If the observed SR is below the benchmark, the formula is mathematically 
    # undefined for a 'minimum' length (you'll never reach it).
    if sr_obs <= sr_bench:
        return np.inf

    # Break down the formula parts
    adjusted_variance = (1 - skew * sr_obs + ((kurt - 1) / 4) * (sr_obs**2))
    precision_required = (z_alpha / (sr_obs - sr_bench))**2
    
    min_trl_obs = 1 + adjusted_variance * precision_required
    
    return {
        'min_trl_observations': int(np.ceil(min_trl_obs)),
        'actual_observations': n_obs,
        'is_statistically_significant': n_obs >= min_trl_obs,
        'annualized_sr': round(sr_obs * np.sqrt(periods_per_year), 2),
        'skew': round(skew, 2),
        'kurtosis': round(kurt, 2)
    }

# ==============================================================================
# MOCK TEST SCENARIO
# ==============================================================================
if __name__ == "__main__":
    print("--- MinTRL Mock Implementation Test ---")
    
    # Scenario A: A "Good" Strategy (Moderate SR, Normalish Returns)
    np.random.seed(42)
    returns_a = np.random.normal(0.0001, 0.01, 5000) # 5000 bars
    equity_a = pd.Series(10000 * np.cumprod(1 + returns_a))
    
    # Scenario B: A "Fragile" Strategy (High SR but Negative Skew/High Kurtosis)
    # i.e., "Picking up pennies" - many small wins, one huge tail risk
    returns_b = np.random.normal(0.0005, 0.01, 5000)
    returns_b[np.random.randint(0, 5000, 10)] = -0.15 # Add 10 "Black Swan" events
    equity_b = pd.Series(10000 * np.cumprod(1 + returns_b))

    results_a = calculate_mintrl(equity_a)
    results_b = calculate_mintrl(equity_b)

    print("\n[Strategy A - Normal]")
    print(f"Annualized SR: {results_a['annualized_sr']}")
    print(f"Actual Obs:    {results_a['actual_observations']}")
    print(f"MinTRL needed: {results_a['min_trl_observations']}")
    print(f"Significant?   {results_a['is_statistically_significant']}")

    print("\n[Strategy B - High Tail Risk (Negative Skew)]")
    print(f"Annualized SR: {results_b['annualized_sr']}")
    print(f"Actual Obs:    {results_b['actual_observations']}")
    print(f"MinTRL needed: {results_b['min_trl_observations']}")
    print(f"Significant?   {results_b['is_statistically_significant']}")
