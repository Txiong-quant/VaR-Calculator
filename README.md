# Portfolio Risk Engine: Value at Risk (VaR) & Kupiec Backtesting

An Object-Oriented Python risk engine that calculates 1-day and multi-day Value at Risk (VaR) using both Parametric and Historical Simulation methodologies. It includes an automated Kupiec Proportion of Failures (POF) backtest to statistically validate model calibration at the 99% confidence interval.

## Phase 1: Parametric VaR (Variance-Covariance)
Calculates portfolio risk by assuming a multivariate normal distribution of asset returns. The model extracts the left-tail risk threshold using the standard normal inverse CDF ($Z$-score) and scales to a multi-day horizon using the square root of time rule ($\sqrt{T}$).

**Math Note - Zero Drift Assumption:** Over a 1-day horizon, daily drift ($\mu$) is statistically negligible compared to daily volatility ($\sigma$). I set $\mu = 0$ to isolate pure volatility-driven downside and eliminate sample mean estimation error. 

**Math Note - Matrix Algebra:** Portfolio variance is calculated via the quadratic form:
$$\sigma_p^2 = w^T \Sigma w$$
Where $w$ is the $(N \times 1)$ allocation weight vector and $\Sigma$ is the $(N \times N)$ sample covariance matrix of historical log returns.

## Phase 2: Historical Simulation VaR
A non-parametric approach that applies current portfolio weights to a 3-year historical log-return distribution, extracting the exact loss threshold at the 1st percentile of empirical daily dollar PnL.

**Math Note - Fat Tails:** Parametric VaR consistently underestimates extreme tail risk because real market returns exhibit leptokurtosis (fat tails) and negative skew. This historical simulation bypasses normal distribution assumptions entirely, automatically capturing real-world volatility clustering and liquidity crashes that actually occurred in the 3-year lookback window.

## Phase 3: Kupiec POF Backtest
A statistical framework to validate if the VaR model's actual failure rate matches its expected failure rate ($p = 1 - c$). It models daily breaches as independent Bernoulli trials.

**Math Note - Wilks' Theorem & Chi-Squared:** The test calculates a Likelihood Ratio ($LR$) comparing the maximum likelihood of the observed failure rate against the expected 1% failure rate. By Wilks' Theorem, $-2 \ln(LR)$ asymptotically converges to a Chi-Squared distribution with 1 degree of freedom ($\chi^2_1$). If the resulting $p$-value falls below $0.05$, the model is statistically miscalibrated and rejected.

## Dependencies
* `numpy` - Matrix operations and vectorized math
* `pandas` - Time-series log return structuring
* `yfinance` - Historical market data ingestion
* `scipy` - Normal distribution CDFs and Chi-Squared testing for Kupiec validation

## Execution
```python
from risk_engine import PortfolioRiskEngine

engine = PortfolioRiskEngine(
    tickers=['AAPL', 'MSFT', 'GLD', 'TLT'],
    weights=[0.25, 0.25, 0.25, 0.25],
    portfolio_val=1_000_000
)

print(engine.calculate_parametric_var(horizon_days=10))
print(engine.calculate_historical_var(horizon_days=10))
print(engine.run_kupiec_backtest())
