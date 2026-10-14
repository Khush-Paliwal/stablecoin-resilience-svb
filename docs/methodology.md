# Methodology

Every parameter below lives in `src/stablecoin_risk/config.py`; the module that implements each block is named
in its heading.

## 1. Event window and regimes

| Regime | Dates (UTC) | Rationale |
|---|---|---|
| Pre-crisis | 1–9 Mar 2023 | baseline for z-scores and multipliers |
| Peak crisis | 10–13 Mar 2023 | SVB closed Fri 10 Mar; Circle disclosure; USDC low $0.87 Sat 11 Mar; backstop announced Sun 12 Mar |
| Recovery | 14–21 Mar 2023 | after the re-peg |

Charts mark SVB failure (10 Mar 12:00), USDC de-peg (11 Mar 08:00) and re-peg (13 Mar 12:00).

## 2. Data cleaning (`data/cleaning.py`)

Missing one-minute bars are imputed **causally**:

* gap ≤ 5 bars: last observation carried forward (flat O = H = L = C bar);
* 6–30 bars: level forecast of simple exponential smoothing fitted on the previous 1,440 closes;
* > 30 bars: left missing, logged to `long_gaps_dropped.csv`.

Imputed bars carry zero volume and an `imputed` flag. Cross-venue comparisons use **traded, non-imputed** bars
only. Markets with no trades and a frozen price are discarded (`is_active_market`).

## 3. Microstructure (`microstructure.py`)

Rolling windows: 60 minutes for impact measures and 1,440 bars for spread estimators. Zero-volume bars are
dropped before estimation.

| Measure | Definition |
|---|---|
| Amihud (2002) | $\text{ILLIQ}_t = \overline{\lvert r_i\rvert / (P_i V_i)}$ over the window |
| Kyle–Obizhaeva (2016) | $(\sigma^2_t / \sum P_i V_i)^{1/3}$ |
| Roll (1984) | $2\sqrt{\max(0, -\text{Cov}(\Delta P_t, \Delta P_{t-1}))} / P_t$ |
| Corwin–Schultz (2012) | $S = 2(e^\alpha - 1)/(1 + e^\alpha)$, $\alpha = \frac{\sqrt{2\beta}-\sqrt\beta}{3-2\sqrt2} - \sqrt{\frac{\gamma}{3-2\sqrt2}}$ from bars $(t-1, t)$, $\alpha$ floored at 0 |
| Abdi–Ranaldo (2017) | $s = \sqrt{\max\{0, \overline{4(c_{t-1}-\eta_{t-1})(c_{t-1}-\eta_t)}\}}$, $\eta$ = log mid-range |
| Efficiency ratio | $\lvert P_t - P_{t-60}\rvert / \sum_{60}\lvert\Delta P\rvert$ |
| VPIN (Easley et al., 2012) | equal-dollar buckets of (avg daily $ volume)/50; $\lvert\text{signed }\$\rvert / \$$ per bucket, sign = close vs bar midpoint |
| Arbitrage gap | $\lvert P^A_t - P^B_t\rvert / P^B_t \times 10^4$ on common traded minutes |

**Market Quality Index.** For each component $k$, $s_{k,t} = \text{clip}((z_{k,t}+2)/8, 0, 1)$ with $z$
measured against the pre-crisis mean and standard deviation (efficiency enters as
$\text{clip}(\lvert ER_t - \overline{ER}\rvert / 4\sigma_{ER}, 0, 1)$).
$\text{MQI}_t = 100\sum_k w_k (1 - s_{k,t})$ with weights spread 0.30, impact 0.30, efficiency 0.20,
VPIN 0.20 (renormalised if VPIN is unavailable). The SPQI is the same index computed on USDC/USD and USDT/USD.

**Order-book Kyle's λ.** Cont–Kukanov–Stoikov (2014) best-level order-flow imbalance between snapshots is summed
into one-minute bars, and $\Delta \text{mid}_t = a + \lambda\, \text{OFI}_t + \varepsilon_t$ is estimated by
rolling OLS over 1,440 bars.

## 4. Basis and regimes (`basis.py`)

$b_t = (P^{BTC/USD}_t - P^{BTC/Q}_t)/P^{BTC/Q}_t$ per venue. A two-state Markov-switching mean model with
switching variance is fitted to 5-minute samples (in bps); the state with the larger absolute mean is labelled
*stress*.

## 5. Volatility connectedness (`spillover.py`)

* Parkinson volatility $V_t = \lvert\ln(H_t/L_t)\rvert/\sqrt{4\ln2}$; weekdays only; VIX as a level.
* Generalised FEVD (Pesaran–Shin 1998):
  $\theta_{ij} = \sigma_{jj}^{-1}\sum_h (e_i' A_h \Sigma e_j)^2 / \sum_h e_i' A_h\Sigma A_h' e_i$, row-normalised.
  Spillover table and total spillover index as in Diebold–Yilmaz (2012).
* **TVP-VAR(2)** (Koop & Korobilis 2014; Antonakakis & Gabauer 2017): random-walk coefficients filtered with
  $P_{t|t-1} = P_{t-1}/\lambda$, $\lambda = 0.99$; EWMA covariance
  $\Sigma_t = \kappa\Sigma_{t-1} + (1-\kappa)\varepsilon_t\varepsilon_t'$, $\kappa = 0.96$; OLS initialisation on
  the first 10% of the sample; GFEVD at $H = 10$ averaged over time after a 20-day burn-in.

## 6. Peg-deviation regressions (`regressions.py`)

$$y_{e,t} = a + \rho y_{e,t-1} + \beta' f(L_t,K_t) + \gamma \text{VIX}_t + \delta_1 HL_{e,t-1} + \delta_2\ln(1+V_{e,t-1}) + \mu_e + \varepsilon_{e,t}$$

* $y$ = log peg deviation, winsorised at the 0.1% / 99.9% quantiles; lags taken within exchange.
* $L$ = (cash + deposits)/reserves, $K$ = (reserves − supply)/reserves, z-scored within each estimation sample.
* USDC: $f = z(L) + z(K)$ (composite safety). USDT: $z(L)$ and $z(K)$ separately.
* Stress specs add a dummy $D_t$ (VIX ≥ 90th percentile, or 9–20 Mar 2023) and its interactions with the
  balance-sheet terms and centred VIX.
* Errors: Newey–West (7 lags) on the stacked panel as in the paper, plus Driscoll–Kraay (time-grouped HAC) as the
  robustness standard.

## 7. Reserve stress tests (`stress_testing.py`)

Equity cushion $E_t$ = reconstructed reserves − tokens outstanding.

| Scenario | Loss |
|---|---|
| A · Liquidity run | $0.02 \times \max(0,\; 0.30\,\text{Supply} - \text{Cash} - \text{O/N repo})$ |
| B · Flash crash | $0.5\,\text{BTC} + 0.15\,\text{Gold} + \text{T-bills}\times(D/365)\times 0.03$ (Circle: $D$ = 45 days) |
| C · Credit contagion | B $+\;0.10\,\text{Corp. bonds} + 0.25\,\text{Secured loans} + 0.50\,\text{Other inv.}$ |

Headline metric: share of days with $E_t < 0$ over Sep 2022 – Feb 2026, chosen because the daily reconstruction
is noisy.

## 8. Exploratory extensions

* **GENIUS Act event study** (`event_study.py`): hourly realised variance of 5-minute log returns on active
  minutes; hour-of-week deseasonalisation estimated outside ±7 days of events; jump regression with Newey–West
  errors; placebo p-values from 200 random same-hour-of-week dates.
* **BTC/USD vs BTC/USDT** (`volatility_models.py`): VAR(1)/Granger, GARCH(1,1) with Student-t errors on
  basis-point returns, bivariate DCC(1,1) by quasi-maximum likelihood.

## References

Abdi & Ranaldo (2017) *RFS* 30(12) · Amihud (2002) *JFM* 5(1) · Antonakakis & Gabauer (2017) MPRA 78282 ·
Cont, Kukanov & Stoikov (2014) *J. Fin. Econometrics* 12(1) · Corwin & Schultz (2012) *JF* 67(2) ·
Diebold & Yilmaz (2012) *IJF* 28(1) · Easley, López de Prado & O'Hara (2012) *RFS* 25(5) · Engle (2002) *JBES* 20(3) ·
Koop & Korobilis (2014) *EER* 71 · Kyle (1985) *Econometrica* 53(6) · Kyle & Obizhaeva (2016) *Econometrica* 84(4) ·
Parkinson (1980) *J. Business* 53(1) · Pesaran & Shin (1998) *Econ. Letters* 58(1) · Roll (1984) *JF* 39(4)
