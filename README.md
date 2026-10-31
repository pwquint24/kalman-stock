# Kalman Filtering vs. ARMA Models for Stock Forecasting

A comparative framework in Wolfram Language for extracting dynamic trend from stock
prices and forecasting future prices, using two distinct mathematical paradigms:
classical ARMA econometrics and Kalman state-space filtering.

## Overview

Predicting future stock prices and extracting dynamic trends are two foundational
problems in quantitative finance. This project implements and rigorously compares:

1. **ARMA models** (Box–Jenkins) — fit to stationary continuously compounded log
   returns, then projected forward toward the unconditional drift rate.
2. **Kalman filtering** (Local Linear Trend state-space model) — estimates unobserved
   hidden states (trend level and momentum) from noisy sequential observations,
   updating recursively with optimal Bayesian variance reduction.

The research question: how accurately can each model predict the **expected average
price** over an investment horizon (e.g. a 20-day trading month)?

## Mathematical Specification

### ARMA on Log Returns

Prices are $I(1)$ and non-stationary, so we model stationary continuously compounded
log returns instead:

$$
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
$$

An $\text{ARMA}(p, q)$ process for returns is

$$
r_t = c + \sum_{i=1}^{p} \phi_i r_{t-i} + \varepsilon_t + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j}
$$

Reconstructing the price path from multi-step forecasted returns:

$$
P_{T+h} = P_T \exp\left(\sum_{i=1}^{h} r_{T+i}\right)
$$

The expected average future price over horizon $H$:

$$
\bar{P}_{\text{ARMA}} = \frac{1}{H}\sum_{h=1}^{H} P_{T+h}
$$

### Local Linear Trend Kalman Filter

Observation and state-transition equations for unobserved level $\mu_t$ and
slope $\beta_t$:

$$
\begin{aligned}
y_t &= \mu_t + \varepsilon_t, & \varepsilon_t &\sim \mathcal{N}(0, \sigma_\varepsilon^2) \\
\mu_{t+1} &= \mu_t + \beta_t + \eta^\mu_t, & \eta^\mu_t &\sim \mathcal{N}(0, \sigma_\mu^2) \\
\beta_{t+1} &= \beta_t + \eta^\beta_t, & \eta^\beta_t &\sim \mathcal{N}(0, \sigma_\beta^2)
\end{aligned}
$$

Forward extrapolation from the terminal state $(\mu_T, \beta_T)$ is analytic:

$$
P_{\text{Kalman}, T+h} = \mu_T + h\,\beta_T
$$

which yields the expected average price over horizon $H$ in closed form:

$$
\bar{P}_{\text{Kalman}} = \frac{1}{H}\sum_{h=1}^{H}\left(\mu_T + h\,\beta_T\right)
= \mu_T + \frac{H+1}{2}\,\beta_T
$$

## Contents

| File | Description |
| --- | --- |
| `Kalman_ARMA_Stock_Forecasting.org` | Literate source document (primary) |
| `Kalman_ARMA_Stock_Forecasting.nb` | Wolfram notebook |
| `Kalman_ARMA_Stock_Forecasting.vsnb` | VS Code notebook |
| `Kalman_ARMA_Stock_Forecasting.md` | Markdown export |

The document covers:

1. Executive summary and theoretical framework
2. Stock data acquisition and preprocessing (SPY, via `FinancialData`)
3. ARMA model estimation on returns
4. Kalman filtering for dynamic trend extraction
5. Out-of-sample performance comparison (single window)
6. Walk-forward backtesting (rolling window evaluation)
7. Analytical discussion and quantitative insights

## Requirements

- Wolfram Language / Mathematica (Wolfram Engine is sufficient)
- Network access for `FinancialData` price retrieval

To evaluate the document, open the `.nb` notebook, or use `org-babel` in Emacs with a
Wolfram Language backend. The `.org` file begins with a package-import block intended
for `sci-wolfram` users.

## License

Distributed under the GNU General Public License v3. See [LICENSE](LICENSE).
