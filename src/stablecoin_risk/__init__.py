"""stablecoin_risk - research code for *Thin Cushions and False Security*.

Evaluating stablecoin resilience under the GENIUS Act, using the March 2023
Silicon Valley Bank failure (and the USDC de-peg it triggered) as a natural
experiment.

Sub-modules
-----------
data.loaders, data.cleaning, data.collectors, data.quality
    Ingestion of exchange OHLCV, issuer reserves and regression inputs.
microstructure
    Liquidity / spread / price-impact estimators, cross-exchange arbitrage gaps,
    the Market Quality Index (MQI) and order-flow-imbalance Kyle's lambda.
basis
    Cross-quote BTC basis and Markov-switching regime detection.
spillover
    Parkinson volatility, static VAR and TVP-VAR GFEVD connectedness.
regressions
    Peg-deviation panel regressions on issuer balance-sheet fundamentals.
stress_testing
    Basel III-style reserve stress scenarios for Tether and Circle.
"""

__version__ = "1.0.0"
