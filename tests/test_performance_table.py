import numpy as np
import pandas as pd
from typing import List, Optional
from utils import generate_performance_table


def generate_synthetic_granular_dfs(
    n_folds: int = 3,
    pairs: Optional[List[str]] = None,
    seed: Optional[int] = 42
) -> List[pd.DataFrame]:
    """
    Generates a list of synthetic per-fold DataFrames matching the structure
    expected by generate_performance_table: each DataFrame is indexed by
    currency pair, with columns being the raw metric keys from metric_meta.
    """
    if pairs is None:
        pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CHF', 'USD/CAD', 'NZD/USD', 'USD/CNH', 'USD/MXN', 'USD/TRY', 'USD/ZAR']

    rng = np.random.default_rng(seed)

    # Plausible (mean, std) per metric, in raw units (e.g. 0.05 = 5% for pct metrics)
    metric_params = {
        'total_return':          (0.08, 0.06),
        'sharpe':                 (1.1, 0.5),
        'probabilistic_sharpe':   (0.65, 0.15),
        'max_dd':                 (-0.12, 0.05),
        'cagr':                   (0.07, 0.04),
        'win_rate':               (0.55, 0.08),
        'profit_factor':          (1.3, 0.3),
        'n_trades':               (150, 40),
        'avg_capital_exposure':   (45, 10),
        'avg_trade_size':         (20, 5),
        'deflated_sharpe':        (0.55, 0.2),
        'm2_brier':               (0.18, 0.05),
    }

    granular_dfs = []
    for fold_idx in range(n_folds):
        data = {}
        for metric, (mean, std) in metric_params.items():
            values = rng.normal(loc=mean, scale=std, size=len(pairs))
            if metric == 'n_trades':
                values = np.round(np.abs(values)).astype(int)
            elif metric in ('win_rate', 'probabilistic_sharpe', 'deflated_sharpe'):
                values = np.clip(values, 0, 1)
            elif metric == 'profit_factor':
                values = np.clip(values, 0.1, None)
            data[metric] = values

        df = pd.DataFrame(data, index=pairs)
        df.index.name = 'Pair'
        granular_dfs.append(df)

    return granular_dfs


if __name__ == "__main__":
    # Example usage
    synthetic_dfs = generate_synthetic_granular_dfs(n_folds=2)
    generate_performance_table(synthetic_dfs, model_name="synthetic_model", phase_name="test")