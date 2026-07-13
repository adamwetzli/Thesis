import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import Optional, Tuple, List
from sklearn.metrics import brier_score_loss

# ==============================================================================
# STANDALONE FINANCIAL STATISTICS ENGINE
# ==============================================================================
# These functions are decoupled from the backtester class to allow for 
# cross-architecture comparison and meta-analysis of results from any source.
# They are designed to work with standard pandas Series for high interoperability.

def estimate_pi0(p_values, lambdas):
    """
    Step 3: Estimate the proportion of null hypotheses (pi0) for a grid of lambdas.
    Formula: pi0(lambda) = #{p_i > lambda} / (n * (1 - lambda))
    """
    n = len(p_values)
    pi0_estimates = []
    for l in lambdas:
        num_above = np.sum(p_values > l)
        pi0 = num_above / (n * (1 - l)) # 
        pi0_estimates.append(min(pi0, 1.0)) # pi0 cannot exceed 100%
    return np.array(pi0_estimates)


def sharpe_ratio_(equity_series, periods_per_year=252*24):
    """
    Calculates the Annualized Sharpe Ratio based on fractional equity returns.
    
    The Sharpe Ratio measures the risk-adjusted return. By using MTM (Mark-to-Market) 
    equity, this calculation faithfully represents the 'path' of the account, 
    including drawdown volatility experienced while trades are open.

    Calculation Logic:
    1. Compute percentage returns bar-by-bar.
    2. Annualize the mean return by the frequency (periods_per_year).
    3. Annualize the volatility (std dev) by the square root of frequency.
    4. Return the ratio of annualized return to annualized volatility.

    Args:
        equity_series (pd.Series): Time-series of account equity (Cash + Floating PnL).
        periods_per_year (int): Annualization factor (e.g., 252 days * 24 hours for hourly data).

    Returns:
        float: The annualized Sharpe Ratio. Returns -6.0 (penalty) for invalid series to 
               guide optimization algorithms away from bankruptcy states.
    """
    if len(equity_series) < 2 or (equity_series <= 0).any():
        return -6.0  # Optimization penalty for accounts that hit 0 or have no history
    
    # Calculate arithmetic returns (percentage change between bars)
    returns = equity_series.pct_change().dropna()
    
    # Handle the case of zero volatility (e.g., no trades or static equity)
    if returns.std() == 0:
        return 0.0
    
    # Formula: (Mean Return / Std Dev of Return) * sqrt(Time)
    # This is mathematically equivalent to (Annualized Return / Annualized Volatility)
    return round((returns.mean() / returns.std()) * np.sqrt(periods_per_year), 2)


def probabilistic_sharpe_ratio_(equity_series, benchmark_sharpe=0.0, periods_per_year=252*24):
    """
    Calculates the Probabilistic Sharpe Ratio (PSR).
    
    PSR accounts for the non-normality of returns (skewness and kurtosis). 
    It provides the probability that the true Sharpe Ratio is greater than a benchmark.
    Developed by Marcos Lopez de Prado.

    Args:
        equity_series (pd.Series): Time-series of account equity.
        benchmark_sharpe (float): Annualized benchmark Sharpe Ratio (e.g., 0.0).
        periods_per_year (int): Annualization factor.

    Returns:
        float: Probability (0.0 to 1.0).
    """
    if len(equity_series) < 3 or (equity_series <= 0).any():
        return 0.0

    returns = equity_series.pct_change().dropna()
    if returns.std() == 0:
        return 0.0

    # 1. Observed non-annualized Sharpe
    n = len(returns)
    sharpe = returns.mean() / returns.std()
    
    # 2. Benchmark non-annualized Sharpe
    sr_benchmark = benchmark_sharpe / np.sqrt(periods_per_year)

    # 3. Calculate Skewness and Kurtosis (Higher order moments)
    skew = returns.skew()
    kurt = returns.kurtosis() + 3 # PSR uses Fisher Kurtosis (usually pandas adds 3, check)
    # Note: pandas .kurtosis() returns excess kurtosis (Fisher), so we add 3 for Pearson.

    # 4. Standard Deviation of the Sharpe Ratio Estimate
    # Formula from Lopez de Prado (2018)
    sigma_sr = np.sqrt((1 - skew * sharpe + (kurt - 1) / 4 * sharpe**2) / (n - 1))

    # 5. PSR (CDF of the Standard Normal distribution)
    psr = norm.cdf((sharpe - sr_benchmark) / sigma_sr)

    return round(float(psr), 4)
    

def max_drawdown_(equity_series):
    """
    Calculates the Maximum Drawdown (MDD) as a percentage.
    
    MDD is the maximum peak-to-trough decline in the account's history. 
    It is the primary measure of absolute risk and "pain" for an investor.
    Since we use MTM equity, this captures the worst point even if it 
    happened mid-trade.

    Args:
        equity_series (pd.Series): Time-series of account equity.

    Returns:
        float: The maximum drawdown as a decimal (e.g., -0.15 for 15% decline).
    """
    if len(equity_series) < 2:
        return 0.0
    
    # Calculate the rolling peak (the highest value seen so far)
    peak = equity_series.cummax()
    
    # Calculate the percentage drop from the highest peak seen to the current value
    drawdown = (equity_series - peak) / peak
    
    # Return the minimum (most negative) value found in the drawdown series
    return round(drawdown.min(), 4)


def cagr_(equity_series, periods_per_year=252*24):
    """
    Calculates the Compound Annual Growth Rate (CAGR).
    
    Unlike simple average returns, CAGR represents the geometric mean return 
    that would provide the same final equity from the initial capital over time, 
    accounting for the compounding effect.

    Args:
        equity_series (pd.Series): Time-series of account equity.
        periods_per_year (int): The number of data bars that constitute one trading year.

    Returns:
        float: The annualized growth rate as a decimal.
    """
    if len(equity_series) < 2 or equity_series.iloc[0] <= 0:
        return 0.0

    initial_equity = equity_series.iloc[0]
    final_equity = equity_series.iloc[-1]
    
    # Convert the number of elapsed bars to a fractional number of years
    total_periods = len(equity_series)
    years = total_periods / periods_per_year

    # Check for invalid timeframes or total capital loss
    if years <= 0 or final_equity <= 0:
        return 0.0
    
    # Standard CAGR Formula: (Final / Initial)^(1 / Years) - 1
    total_return = final_equity / initial_equity
    cagr = np.power(total_return, 1 / years) - 1

    return round(cagr, 4)


def expected_max_sr_(num_trials, variance_of_sharpes):
    """
    Calculates the Expected Maximum Sharpe Ratio (SR*) under the Null Hypothesis.
    Uses the Euler-Mascheroni constant approximation for the expected maximum of 
    N independent standard normal variables.
    """
    import math
    if num_trials <= 1:
        return 0.0
    
    gamma = 0.5772156649
    num_trials = max(num_trials, 2)
    z_inv_term1 = norm.ppf(1.0 - (1.0 / num_trials))
    z_inv_term2 = norm.ppf(1.0 - (1.0 / (num_trials * math.exp(1))))
    
    # Expected Max Sharpe (Annualized)
    # sr_star = sqrt(V) * [(1 - gamma) * Z^-1[1 - 1/N] + gamma * Z^-1[1 - 1/(N*e)]]
    return np.sqrt(variance_of_sharpes) * ((1 - gamma) * z_inv_term1 + gamma * z_inv_term2)


def deflated_sharpe_ratio_(equity_series, num_trials, variance_of_sharpes, periods_per_year=252*24):
    """
    Calculates the Deflated Sharpe Ratio (DSR).
    
    DSR adjusts the Sharpe Ratio for multiple testing (selection bias). 
    It is the PSR calculated against a benchmark that accounts for the 
    expected maximum Sharpe Ratio found by chance across many trials.
    Developed by Marcos Lopez de Prado.

    Args:
        equity_series (pd.Series): Time-series of account equity.
        num_trials (int): The number of independent trials (parameter combinations) tested.
        variance_of_sharpes (float): The variance of the Sharpe Ratios observed across all trials.
        periods_per_year (int): Annualization factor.

    Returns:
        float: The probability (0.0 to 1.0) that the true SR is positive after 
               accounting for selection bias.
    """
    sr_star = expected_max_sr_(num_trials, variance_of_sharpes)
    return probabilistic_sharpe_ratio_(equity_series, benchmark_sharpe=sr_star, periods_per_year=periods_per_year)


def calculate_mintrl_(equity_series, benchmark_sharpe=0.0, alpha=0.05, periods_per_year=252*24):
    """
    Calculates the Minimum Track Record Length (MinTRL).
    
    MinTRL provides the number of observations required to reject the null 
    hypothesis that the true Sharpe Ratio is less than or equal to a benchmark.
    It accounts for the non-normality (skewness and kurtosis) of the returns.

    Args:
        equity_series (pd.Series): Time-series of account equity.
        benchmark_sharpe (float): Annualized benchmark Sharpe Ratio (default 0.0).
        alpha (float): Significance level (default 0.05 for 95% confidence).
        periods_per_year (int): Annualization factor.

    Returns:
        int: The required number of observations (bars). Returns np.inf if 
             the observed Sharpe Ratio is below the benchmark.
    """
    if len(equity_series) < 4:
        return np.inf

    returns = equity_series.pct_change().dropna()
    if returns.std() == 0:
        return np.inf

    # 1. Observed non-annualized Sharpe
    sharpe = returns.mean() / returns.std()
    
    # 2. Benchmark non-annualized Sharpe
    sr_benchmark = benchmark_sharpe / np.sqrt(periods_per_year)
    
    # If the observed SR is below the benchmark, the goal is unreachable
    if sharpe <= sr_benchmark:
        return np.inf

    # 3. Calculate Higher Moments
    skew = returns.skew()
    kurt = returns.kurtosis() + 3 
    
    # 4. Z-score for the target confidence
    z_alpha = norm.ppf(1 - alpha)
    
    # 5. MinTRL Formula
    # MinTRL = 1 + [1 - skew*SR + ((kurt-1)/4)*SR^2] * (Z_alpha / (SR - SR_bench))^2
    adj_variance = (1 - skew * sharpe + ((kurt - 1) / 4) * (sharpe**2))
    min_trl = 1 + adj_variance * (z_alpha / (sharpe - sr_benchmark))**2
    
    return int(np.ceil(min_trl))


def win_rate_(trade_history):
    """
    Calculates the strategy win rate based on Net Round-Trip PnL.
    
    A "Win" is strictly defined as a trade where the price gain exceeds 
    the sum of both ENTRY and EXIT transaction costs and slippage.

    Args:
        trade_history (list): List of dictionaries, each containing 'net_pnl'.

    Returns:
        float: Decimal win rate (0.0 to 1.0).
    """
    if not trade_history:
        return 0.0
    
    # Count trades where the net impact on account balance was positive
    winning_trades = [t for t in trade_history if t.get('net_pnl', 0) > 0]
    return round(len(winning_trades) / len(trade_history), 4)


def avg_capital_exposure_(exposure_history):
    """
    Calculates the Time-Weighted Average Capital Exposure (%).
    Includes bars where the account was in cash (0% exposure).
    """
    if not exposure_history:
        return 0.0
    return round(np.mean(exposure_history), 2)


def avg_trade_size_(exposure_history):
    """
    Calculates the Average Trade Size (%) only when a position was open.
    Excludes bars where exposure was 0.
    """
    active_exposure = [e for e in exposure_history if e > 0]
    if not active_exposure:
        return 0.0
    return round(np.mean(active_exposure), 2)


def profit_factor_(trade_history):
    """
    Calculates the Profit Factor (Gross Gains / Gross Losses).
    
    The Profit Factor is a measure of the strategy's 'payoff robustness'. 
    A value > 1.0 indicates profitability. In institutional systems, a 
    value > 1.5 is often the minimum threshold for production readiness.

    Args:
        trade_history (list): List of dictionaries containing 'net_pnl'.

    Returns:
        float: The profit factor. Returns np.inf if there are zero losses.
    """
    if not trade_history:
        return 0.0
    
    # Aggregate all winning trades and losing trades separately
    gains = [t['net_pnl'] for t in trade_history if t['net_pnl'] > 0]
    losses = [abs(t['net_pnl']) for t in trade_history if t['net_pnl'] < 0]
    
    sum_gains = sum(gains)
    sum_losses = sum(losses)
    
    # Handle the 'holy grail' scenario of zero losses
    if sum_losses == 0:
        return np.inf if sum_gains > 0 else 0.0
        
    return round(sum_gains / sum_losses, 2)


# ==============================================================================
# CORE BACKTESTING ENGINE
# ==============================================================================

class CustomBacktester:
    """
    A professional, event-driven backtesting engine designed for the 
    Triple Barrier Method (TBM) and Meta-Labeling architectures.

    DISTINGUISHING FEATURES:
    -----------------------
    1. Mark-to-Market (MTM) Tracking: 
       Updates portfolio equity at every single bar. This includes 'floating PnL' 
       (unrealized profit/loss of open positions), which ensures that Sharpe Ratio 
       and Drawdown metrics reflect real-world volatility and path-dependency.
       
    2. Symmetric Barriers: 
       Supports absolute price distances (pips/points) for SL and TP. This 
       eliminates the mathematical 'Percentage Asymmetry Bias' where short 
       trades are structurally harder to win than long trades.
       
    3. Institutional Risk Management: 
       - Risk Parity: Dynamically sizes positions based on a fixed % of equity risk.
       - M2 Confidence Scaling: Multiplies trade size by the conviction of the 
         Meta-Model (M2), using the 'Edge' logic (2P - 1).
       - Fractional Kelly: Applies a safety multiplier (e.g., 0.5 for Half-Kelly) 
         to protect against model estimation errors.
         
    4. High-Fidelity Cost Model: 
       Deducts transaction costs (commissions) and slippage bar-by-bar to 
       maintain an accurate realized cash balance.
    """

    def __init__(self, 
                 ohlc: pd.DataFrame, 
                 signals: pd.Series, 
                 initial_cash: float = 10000.0,
                 tc_per_unit: float = 0.0001,
                 slippage_per_unit: float = 0.0002,
                 max_notional_exposure_pct: float = 100.0): # NEW PARAMETER
        """
        Initializes the backtesting environment.

        Args:
            ohlc (pd.DataFrame): Historical price data. Must contain ['high', 'low', 'close'].
            signals (pd.Series): Directional signals from the Primary Model (M1). 
                                -1=Short, 0=Neutral, 1=Long.
            initial_cash (float): Starting balance for the simulation.
            tc_per_unit (float): Transaction cost (commission) per contract/unit traded.
            slippage_per_unit (float): Average expected slippage (price degradation) per unit.
        """
        self.ohlc = ohlc
        self.signals = signals
        self.initial_cash = initial_cash
        self.equity = initial_cash # Tracks the 'Realized Cash' in the account
        self.tc_per_unit = tc_per_unit
        self.slippage_per_unit = slippage_per_unit
        self.max_notional_exposure_pct = max_notional_exposure_pct # <-- STORED
        
        # Performance logging buffers
        self.trade_history = []        # Stores details of every closed trade
        self.equity_history = []       # Tracks MTM equity at every bar for statistics
        self.exposure_history = []     # Tracks what % of account was deployed at each bar (Exposure)
        self.open_trades = []          # List of currently active positions
        
        # Cumulative performance counters
        self.total_tc = 0
        self.total_costs = 0
        self.total_pnl = 0

    def run(self, 
            sl: float | pd.Series, 
            tp: float | pd.Series, 
            max_holding_periods: int, 
            risk_pct: float = 0.01, 
            conformal_confidence: pd.Series = None,
            significance: float = 0.1,
            kelly_fraction: float = 0.5,
            min_qty: float = 1.0,
            max_qty: float = 100000.0,
            is_distance: bool = False):
        """
        Executes the main backtest simulation loop over the historical data.

        OPERATIONAL WORKFLOW (Per Bar):
        -----------------------------
        1. EXIT CHECK: Iterates through open trades and checks if the High/Low 
           prices hit the Stop-Loss or Take-Profit barriers, or if the time 
           limit (Timeout) or an opposing signal occurred.
           
        2. ENTRY CHECK: If the account is neutral (no open trades) and a signal 
           exists, it evaluates the entry.
           
        3. CONFORMAL POSITION SIZING: 
           - Calculates how many units to trade to lose exactly 'risk_pct' of equity if SL hit.
           - Adjusts this size based on Excess Conformal Confidence.
           - Applies Kelly scaling to optimize for long-term growth vs safety.
           
        4. ACCOUNTING: Calculates MTM equity by adding current floating PnL to cash.

        Args:
            sl: Stop-Loss value. Generic to support fixed % or dynamic ATR distance.
            tp: Take-Profit value. Generic to support fixed % or dynamic ATR distance.
            max_holding_periods (int): The "Vertical Barrier" (Timeout) for TBM.
            risk_pct (float): Percent of current equity to risk per trade (Risk Parity).
            conformal_confidence (pd.Series): (1 - p_value) series from Meta-Model.
            significance (float): Conformal significance level (the trading threshold).
            kelly_fraction (float): Multiplier for Kelly scaling (0.5 = Half-Kelly).
            min_qty/max_qty: Liquidity and safety constraints on position size.
            is_distance (bool): Set to True if sl/tp are provided as ATR price gaps.
        """
        for i in range(len(self.ohlc)):
            timestamp = self.ohlc.index[i]
            current_price = self.ohlc['close'].iloc[i]
            current_low = self.ohlc['low'].iloc[i]
            current_high = self.ohlc['high'].iloc[i]
            current_signal = self.signals.iloc[i]

            # --- 1. BARRIER CHECK (EXIT LOGIC) ---
            if self.open_trades:
                for open_trade in self.open_trades[:]:
                    exit_reason = None
                    exit_price = None

                    # A. Evaluate Horizontal Barriers (Symmetric)
                    # For Longs: Triggered if Low drops to SL or High reaches TP.
                    if open_trade['direction'] == 'long':
                        if current_low <= open_trade['sl_price']:
                            exit_reason = 'Stop-Loss'
                            exit_price = open_trade['sl_price']
                        elif current_high >= open_trade['tp_price']:
                            exit_reason = 'Take-Profit'
                            exit_price = open_trade['tp_price']
                    
                    # For Shorts: Triggered if High rises to SL or Low reaches TP.
                    else: # Short position
                        if current_high >= open_trade['sl_price']:
                            exit_reason = 'Stop-Loss'
                            exit_price = open_trade['sl_price']
                        elif current_low <= open_trade['tp_price']:
                            exit_reason = 'Take-Profit'
                            exit_price = open_trade['tp_price']

                    # B. Evaluate Vertical Barrier (Timeout)
                    # Exits the trade if it has been open for too many periods.
                    if not exit_reason and (i - open_trade['entry_idx']) >= max_holding_periods:
                        exit_reason = 'Timeout'
                        exit_price = current_price

                    # C. Evaluate Opposing Signal (Veto Exit)
                    # Exits if the Primary Model (M1) issues a signal in the opposite direction.
                    if not exit_reason and \
                       ((open_trade['direction'] == 'long' and current_signal == -1) or \
                        (open_trade['direction'] == 'short' and current_signal == 1)):
                        exit_reason = 'Opposing Signal'
                        exit_price = current_price

                    # Finalize trade closure if any reason was found
                    if exit_reason:
                        self._close_trade(open_trade, timestamp, exit_price, exit_reason)

            # --- 2. SIGNAL EVALUATION (ENTRY LOGIC) ---
            # Only enters if no trades are currently open (Single-position logic)
            if not self.open_trades and current_signal != 0:
                # Resolve current SL/TP barriers (Supports dynamic bar-by-bar Series)
                val_sl = sl.iloc[i] if isinstance(sl, pd.Series) else sl
                val_tp = tp.iloc[i] if isinstance(tp, pd.Series) else tp
                
                # Determine absolute price distance for sizing and barriers.
                # Distance (e.g. 0.0050) is used for symmetric risk parity.
                dist_sl = val_sl if is_distance else (current_price * val_sl)
                dist_tp = val_tp if is_distance else (current_price * val_tp)

                # --- CONFORMAL POSITION SIZING ---
                
                # LAYER 1: Risk Parity (Normalization)
                # Calculates base size such that hitting 'sl' results in exactly 'risk_pct' loss.
                # Formula: Base Size = (Current Capital * Risk%) / Pip Distance to Stop
                base_qty = (self.equity * risk_pct) / dist_sl if dist_sl != 0 else min_qty

                # LAYER 2: Conformal Edge Scaling (Excess Confidence Logic)
                # Confidence C = 1 - p_value. Threshold T = 1 - Significance.
                if conformal_confidence is not None:
                    conf = conformal_confidence.iloc[i]
                    threshold = 1.0 - significance

                    # Calculate the "Z-score" of your confidence
                    # We treat 'significance' as the scale (the 'sigma')
                    if significance > 0:
                        # This is our Signal-to-Noise ratio
                        z = (conf - threshold) / significance
                    else:
                        z = 0

                    # Apply the Lopez de Prado Sigmoid (Gaussian CDF)
                    # norm.cdf(z) gives the probability from -inf to z
                    # 2*cdf - 1 maps it to the range [-1, 1]. We clamp it at 0.
                    size = max(0, 2 * norm.cdf(3*z) - 1)
                else:
                    size = 1.0 # Assume full conviction if no confidence data provided

                # LAYER 3: Fractional Kelly (Optimal Growth with Safety)
                final_qty = base_qty * size * kelly_fraction
                
                # NEW GUARDRAIL: Cap final_qty based on max_notional_exposure_pct
                # Use current realized cash (self.equity) as the base for percentage calculation.
                effective_current_equity = self.equity
                max_allowed_notional_value = effective_current_equity * (self.max_notional_exposure_pct / 100.0)

                max_qty_based_on_pct = max_qty # Default to existing max_qty if price is zero or invalid
                if current_price > 0:
                    max_qty_based_on_pct = max_allowed_notional_value / current_price
                else: # If current_price is 0 or negative, effectively disallow trading with this guardrail
                    max_qty_based_on_pct = 0 

                # Apply all quantity limits: min_qty, existing max_qty, and the new percentage-based max.
                final_qty = max(min_qty, min(final_qty, max_qty, max_qty_based_on_pct))

                self._open_trade(i, timestamp, current_price, current_signal, dist_sl, dist_tp, final_qty)
            
            # --- 3. PORTFOLIO ACCOUNTING (MTM LOGIC) ---
            total_invested, floating_pnl, mtm_equity = self._compute_mtm(current_price)

            # --- 3b. HARD EXPOSURE CEILING (MARK-TO-MARKET DE-LEVERAGING) ---
            # The entry-time sizing guardrail only prevents opening a position that is
            # too large *at the moment of entry*. It does NOT protect against exposure
            # drifting above the cap afterwards, e.g. because floating losses shrink
            # mtm_equity while notional exposure (fixed at entry) stays the same.
            # Here we treat max_notional_exposure_pct as an absolute ceiling: if MTM
            # notional exposure ever exceeds it, we forcibly trim the open position(s)
            # down to the boundary, realizing the corresponding PnL/costs immediately.
            if self.open_trades and mtm_equity > 0 and current_price > 0:
                max_allowed_notional_value = mtm_equity * (self.max_notional_exposure_pct / 100.0)

                if total_invested > max_allowed_notional_value:
                    excess_notional = total_invested - max_allowed_notional_value

                    # Only one position is ever open at a time under this strategy's
                    # single-position entry logic, but we loop generically (and
                    # allocate the trim pro-rata by notional) in case that changes.
                    for trade in self.open_trades[:]:
                        if excess_notional <= 1e-9:
                            break
                        trade_notional = abs(trade['size']) * trade['entry_price_slippage']
                        if trade_notional <= 0:
                            continue
                        share = trade_notional / total_invested if total_invested > 0 else 1.0
                        reduce_notional = min(trade_notional, excess_notional * share)
                        reduce_qty = min(abs(trade['size']), reduce_notional / current_price)

                        if reduce_qty > 0:
                            self._reduce_trade(trade, timestamp, current_price, reduce_qty, 'Exposure Cap (De-lever)')
                            excess_notional -= reduce_notional

                    # Re-derive MTM state now that the position(s) have been trimmed so
                    # exposure_history/equity_history reflect the post-de-leverage reality.
                    total_invested, floating_pnl, mtm_equity = self._compute_mtm(current_price)

            # Exposure analysis: tracking what % of the account was 'at risk' at this bar
            exposure_pct = (total_invested / mtm_equity) * 100 if mtm_equity > 0 else 0
            self.exposure_history.append(exposure_pct)

            # Record the full portfolio state for downstream Sharpe/CAGR/MDD calculation.
            # This ensures that the statistics see the 'full path' of the backtest.
            self.equity_history.append({
                'timestamp': timestamp,
                'equity': mtm_equity
            })
        
        return self.get_stats()

    def _compute_mtm(self, current_price):
        """
        Computes total notional exposure, floating PnL, and MTM equity for the
        current set of open trades at the given mark price. Factored out so the
        exposure-ceiling guardrail can cheaply re-derive these values after
        trimming a position, without duplicating the accumulation logic.
        """
        total_invested = 0
        floating_pnl = 0
        for trade in self.open_trades:
            # Track total notional capital currently deployed in the market
            total_invested += abs(trade['size']) * trade['entry_price_slippage']

            # Calculate the floating (unrealized) PnL of active trades relative to current price.
            # This must account for the slippage paid at entry.
            if trade['direction'] == 'long':
                floating_pnl += (current_price - trade['entry_price_slippage']) * abs(trade['size'])
            else:
                floating_pnl += (trade['entry_price_slippage'] - current_price) * abs(trade['size'])

        # MTM Equity = Realized Cash Balance (Available) + Floating Unrealized PnL (Open)
        mtm_equity = self.equity + floating_pnl
        return total_invested, floating_pnl, mtm_equity

    def _reduce_trade(self, trade, timestamp, exit_price, reduce_qty, exit_reason):
        """
        Partially closes an open trade by `reduce_qty` units at the current market
        price, realizing PnL/costs on just that slice while leaving the remainder
        of the position open. Used by the exposure-ceiling guardrail to forcibly
        de-lever a position whose MTM notional has drifted above the configured
        max_notional_exposure_pct, without disturbing the rest of the trade's
        lifecycle (SL/TP/timeout logic continues to apply to the remaining size).

        If reduce_qty consumes the entire remaining position, the trade is closed
        out fully via the normal _close_trade path instead (so it is removed from
        open_trades and logged consistently with other full exits).
        """
        direction = trade['direction']
        reduce_qty = min(reduce_qty, abs(trade['size']))
        if reduce_qty <= 0:
            return

        # If this reduction would close out the whole remaining position, just
        # route it through the standard full-close path.
        if reduce_qty >= abs(trade['size']) - 1e-9:
            self._close_trade(trade, timestamp, exit_price, exit_reason)
            return

        # Cost of exiting just the `reduce_qty` slice
        exit_tc = self.tc_per_unit * reduce_qty
        exit_slippage = self.slippage_per_unit * reduce_qty
        total_exit_cost = exit_tc + exit_slippage

        # Effective exit price for this slice (degraded by slippage, same convention
        # as a full close: longs sell into the bid, shorts buy back at the ask).
        if direction == 'long':
            effective_exit = exit_price - self.slippage_per_unit
        else:
            effective_exit = exit_price + self.slippage_per_unit

        # Price PnL on just the reduced quantity
        if direction == 'long':
            price_pnl = (effective_exit - trade['entry_price_slippage']) * reduce_qty
        else:
            price_pnl = (trade['entry_price_slippage'] - effective_exit) * reduce_qty

        # Realized cash impact: price PnL on the closed slice minus its exit commission.
        realized_pnl = price_pnl - exit_tc
        self.equity += realized_pnl

        # Proportional entry commission attributable to this slice (tc_per_unit is a
        # constant rate, so this is exact, not an approximation).
        entry_tc_for_slice = self.tc_per_unit * reduce_qty
        round_trip_net_pnl = price_pnl - entry_tc_for_slice - exit_tc

        # Bookkeeping (entry-side costs for this slice were already deducted from
        # self.equity/counted in total_costs at trade open; only add the exit side here).
        self.total_costs += total_exit_cost
        self.total_tc += exit_tc
        self.total_pnl += round_trip_net_pnl

        # Log the partial exit as its own record so trade-level stats (win rate,
        # profit factor, trade log) reflect the realized de-leveraging event.
        self.trade_history.append({
            'entry_time': trade['entry_time'],
            'exit_time': timestamp,
            'direction': direction,
            'size': reduce_qty if direction == 'long' else -reduce_qty,
            'entry_price_raw': trade['entry_price_raw'],
            'entry_price_slippage': trade['entry_price_slippage'],
            'exit_price_raw': exit_price,
            'exit_price_slippage': effective_exit,
            'net_pnl': round_trip_net_pnl,
            'exit_reason': exit_reason
        })

        # Shrink the remaining open position by the reduced quantity, preserving sign.
        remaining_qty = abs(trade['size']) - reduce_qty
        trade['size'] = remaining_qty if direction == 'long' else -remaining_qty

    def _open_trade(self, entry_idx, timestamp, entry_price, signal, dist_sl, dist_tp, size):
        """
        Internal method to execute a market entry, calculate symmetric barriers, 
        and initialize trade state tracking.
        """
        direction = 'long' if signal == 1 else 'short'
        
        # Calculate upfront execution costs (Transaction Fees + Market Impact/Slippage)
        tc_cost = self.tc_per_unit * abs(size)
        slippage = self.slippage_per_unit * abs(size)
        total_entry_cost = tc_cost + slippage

        # Determine effective entry price based on direction:
        # Longs buy at the High (Ask) side, Shorts sell at the Low (Bid) side.
        if direction == 'long':
            effective_entry = entry_price + self.slippage_per_unit
        else:
            effective_entry = entry_price - self.slippage_per_unit

        # Calculate Symmetric Barrier prices using the absolute price gaps.
        # This ensures perfectly identical 'difficulty' for Long/Short hits.
        if direction == 'long':
            sl_price = effective_entry - dist_sl
            tp_price = effective_entry + dist_tp
        else:
            sl_price = effective_entry + dist_sl
            tp_price = effective_entry - dist_tp

        # Package the trade metadata for the barrier monitoring loop
        trade = {
            'entry_idx': entry_idx,
            'entry_time': timestamp,
            'entry_price_raw': entry_price,
            'entry_price_slippage': effective_entry,
            'size': size if direction == 'long' else -size,
            'direction': direction,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'entry_costs': total_entry_cost,
            'entry_tc': tc_cost # Entry commissions are deducted from cash balance immediately
        }
        
        # Real-time accounting: entry transaction costs are deducted from realized cash immediately.
        self.equity -= tc_cost 
        self.total_costs += total_entry_cost
        self.total_tc += tc_cost
        self.open_trades.append(trade)

    def _close_trade(self, trade, timestamp, exit_price, exit_reason):
        """
        Internal method to finalize a trade, calculate round-trip Net PnL, 
        and update the realized account balance.
        """
        direction = trade['direction']
        
        # Calculate market exit costs (TC and Slippage)
        exit_tc = self.tc_per_unit * abs(trade['size'])
        exit_slippage = self.slippage_per_unit * abs(trade['size'])
        total_exit_cost = exit_tc + exit_slippage

        # Determine effective exit price (Market degraded by slippage)
        if direction == 'long':
            effective_exit = exit_price - self.slippage_per_unit
        else:
            effective_exit = exit_price + self.slippage_per_unit

        # Calculate the Pure Price PnL (excluding commissions)
        if direction == 'long':
            price_pnl = (effective_exit - trade['entry_price_slippage']) * abs(trade['size'])
        else:
            price_pnl = (trade['entry_price_slippage'] - effective_exit) * abs(trade['size'])

        # Real-time Accounting: realized cash is updated by price PnL minus the exit commissions.
        # (Entry commissions were already deducted at _open_trade).
        realized_pnl = price_pnl - exit_tc
        self.equity += realized_pnl
        
        # Round-Trip Net PnL (The final 'bottom line' for this specific trade).
        # Used for accurate Win-Rate and Profit Factor reporting.
        # Formula: Gross Price Result - Total Commissions (Entry + Exit)
        round_trip_net_pnl = price_pnl - trade['entry_tc'] - exit_tc

        # Accumulate global performance logs for final audit
        self.total_costs += (total_exit_cost + trade['entry_costs'])
        self.total_tc += (exit_tc + trade['entry_tc'])
        self.total_pnl += round_trip_net_pnl

        # Log final trade results to history
        self.trade_history.append({
            'entry_time': trade['entry_time'],
            'exit_time': timestamp,
            'direction': direction,
            'size': trade['size'],
            'entry_price_raw': trade['entry_price_raw'],
            'entry_price_slippage': trade['entry_price_slippage'],
            'exit_price_raw': exit_price,
            'exit_price_slippage': effective_exit,
            'net_pnl': round_trip_net_pnl, 
            'exit_reason': exit_reason
        })
        self.open_trades.remove(trade)

    def get_stats(self, num_trials: Optional[int] = None, variance_of_sharpes: Optional[float] = None, p_values: Optional[List] = None,
                  brier_tuple: Optional[Tuple] = None):
        """
        Computes the final performance dictionary for the full backtest simulation.
        Utilizes the standalone stats engine for consistent metric calculation.
        """
        equity_df = pd.DataFrame(self.equity_history).set_index('timestamp')
        if equity_df.empty: return {}
        
        equity_series = equity_df['equity']
        sr_obs = sharpe_ratio_(equity_series)
        
        stats = {
            'total_return': (equity_series.iloc[-1] / self.initial_cash) - 1,
            'sharpe': sr_obs,
            'probabilistic_sharpe': probabilistic_sharpe_ratio_(equity_series),
            'min_trl': calculate_mintrl_(equity_series),
            'max_dd': max_drawdown_(equity_series),
            'cagr': cagr_(equity_series),
            'win_rate': win_rate_(self.trade_history),
            'profit_factor': profit_factor_(self.trade_history),
            'n_trades': len(self.trade_history),
            'avg_capital_exposure': avg_capital_exposure_(self.exposure_history),
            'avg_trade_size': avg_trade_size_(self.exposure_history),
            'portfolio_df': equity_df,
            'trade_history': self.trade_history
        }

        # Calculate DSR and Deflated MinTRL if trial metadata is provided
        if num_trials is not None and variance_of_sharpes is not None:
            sr_star = expected_max_sr_(num_trials, variance_of_sharpes)
            stats['deflated_sharpe'] = probabilistic_sharpe_ratio_(
                equity_series, benchmark_sharpe=sr_star, periods_per_year=252*24
            )
            stats['deflated_min_trl'] = calculate_mintrl_(
                equity_series, benchmark_sharpe=sr_star, periods_per_year=252*24
            )
        
        if brier_tuple is not None:
            m1_correct = brier_tuple[0]
            probs_m2 = brier_tuple[1]
            stats['m2_brier'] = brier_score_loss(m1_correct, probs_m2)
        
        return stats