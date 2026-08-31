# Machine Learning for Cross-Sectional Return Prediction

A research project on cross-sectional U.S. equity return prediction using
price-derived features, linear and nonlinear machine-learning models,
purged walk-forward evaluation, market-regime diagnostics, and portfolio
backtesting.

The main empirical finding is that simple volatility features produced more
robust out-of-sample ranking information than broader momentum/trend feature
sets or a nonlinear gradient-boosting model.

---

## Overview

The project studies whether historical price information can predict the
relative future performance of stocks.

For stock $i$ at date $t$, the prediction target is its future
21-trading-day return:

```math
y_{i,t}
=
\frac{P_{i,t+21}}{P_{i,t}} - 1
```

Models are evaluated primarily using daily cross-sectional Rank IC:

```math
\mathrm{RankIC}_t
=
\mathrm{Corr}
\left(
\mathrm{Rank}(\hat y_{i,t}),
\mathrm{Rank}(y_{i,t})
\right)
```

rather than relying only on point-forecast metrics such as MSE or $R^2$.

---

## Data

The modeling panel contains approximately 199 U.S. equities with daily price
history from 2016 to 2025.

The universe was constructed from current S&P 500 constituents and then
projected backward through historical prices.

This creates an important survivorship / universe-selection bias, which is
explicitly discussed in the limitations section.

Each observation corresponds to:

```text
(date, ticker, features, future_21d_return)
```

so one row represents one stock on one date.

---

## Feature Engineering

The full feature set contains 12 price-derived variables.

### Return and Momentum

- 1-day return
- 5-day return
- 21-day return
- 63-day return
- 126-day return
- 126-day momentum with a 5-day skip

The skip-momentum feature is:

```math
\mathrm{Momentum}_{126,5}
=
\frac{P_{t-5}}{P_{t-126}} - 1
```

### Volatility

- 5-day realized volatility
- 20-day realized volatility
- 63-day realized volatility

For example, 20-day volatility is the rolling standard deviation of daily
returns over the previous 20 trading days.

### Trend and Price Position

- 63-day drawdown from rolling high
- 5-day / 21-day moving-average ratio
- 21-day / 63-day moving-average ratio

The drawdown feature is:

```math
\mathrm{Drawdown}_{63,t}
=
\frac{P_t}
{\max(P_{t-62},\ldots,P_t)}
-1
```

All features use information available at or before the prediction date.

---

## Leakage Control

Random train/test splitting is inappropriate for this time-dependent problem.

The evaluation therefore uses chronological train/test splits.

Because the target itself uses the following 21 trading days, a
21-trading-day purge window is inserted before every test period. This prevents
training labels near the boundary from using prices that belong to the test
period.

The main evaluation uses an expanding-window walk-forward design:

```text
Train through 2021 -> predict 2022

Train through 2022 -> predict 2023

Train through 2023 -> predict 2024

Train through 2024 -> predict 2025
```

Feature standardization is also fitted on training data only:

```text
Train:
fit scaler + transform

Test:
transform only
```

so future distributional information is not used during preprocessing.

---

## Models

### Ridge Regression

Ridge regression serves as the primary linear baseline:

```math
\hat y
=
\beta_0 + \beta^\top x
```

with an L2 penalty:

```math
\lambda \sum_j \beta_j^2
```

Ridge was chosen as a stable baseline because many price-derived features are
strongly correlated.

### Histogram Gradient Boosting

`HistGradientBoostingRegressor` was used as a nonlinear comparison model.

Unlike Ridge, the tree-based model can represent nonlinear thresholds and
feature interactions.

However, greater model complexity did not improve overall out-of-sample
performance in this experiment.

---

## Initial Model Comparison

Using the broader 12-feature specification:

| Model | Mean Rank IC | Median Rank IC | Positive IC Ratio |
|---|---:|---:|---:|
| Ridge | **0.0267** | 0.0047 | 50.97% |
| HistGradientBoosting | 0.0077 | -0.0100 | 47.50% |

The nonlinear model did not outperform the simpler Ridge baseline.

---

## Feature Ablation

Features were separated into three economically interpretable groups.

| Feature Group | Walk-Forward Mean Rank IC |
|---|---:|
| Volatility | **0.0378** |
| All 12 features | 0.0267 |
| Trend / position | 0.0063 |
| Return / momentum | -0.0145 |

The volatility group provided the strongest out-of-sample ranking signal.

Adding more features therefore did not automatically improve prediction.

### Volatility Horizon Test

The volatility group was further decomposed:

| Feature Set | Mean Rank IC |
|---|---:|
| vol_5d + vol_20d + vol_63d | **0.0378** |
| vol_63d | 0.0343 |
| vol_20d | 0.0336 |
| vol_5d | 0.0263 |
| volatility + trend | 0.0234 |

The three volatility horizons contained some complementary information, while
adding trend features reduced performance.

---

## Walk-Forward Stability

![Purged Walk-Forward Rank IC](figures/yearly_rank_ic.png)

Volatility-only Ridge Mean Rank IC by test year:

| Year | Mean Rank IC |
|---|---:|
| 2022 | -0.031 |
| 2023 | 0.075 |
| 2024 | 0.020 |
| 2025 | 0.092 |

The signal was clearly time-varying.

It failed in 2022, then produced positive average Rank IC in each of the
following three test years.

This is an important reminder that a relationship estimated from historical
financial data need not remain stationary.

---

## Volatility Coefficients

Across successive expanding training windows, Ridge assigned positive
coefficients to all three volatility variables.

```text
             vol_5d    vol_20d    vol_63d
Predict 2022    +          +           +
Predict 2023    +          +           +
Predict 2024    +          +           +
Predict 2025    +          +           +
```

The 63-day volatility coefficient was consistently the largest.

This suggests that, within the training samples, the model associated higher
recent volatility with higher expected future raw returns.

However, coefficient stability does not imply that the relationship works in
every future period, as demonstrated by the negative 2022 out-of-sample IC.

---

## Volatility Quintile Check

As a simple diagnostic, stocks were ranked each day by 63-day volatility and
divided into five cross-sectional quintiles.

Average subsequent 21-day returns in the out-of-sample period were:

| Volatility Quintile | Future 21-Day Return |
|---|---:|
| Q1 — lowest volatility | 0.58% |
| Q2 | 0.56% |
| Q3 | 0.67% |
| Q4 | 1.00% |
| Q5 — highest volatility | **1.99%** |

The average Q5-minus-Q1 spread was:

```text
+1.42%
```

and the spread was positive on approximately 57% of evaluated dates.

By year:

| Year | Q5 - Q1 Spread |
|---|---:|
| 2022 | -0.53% |
| 2023 | +2.40% |
| 2024 | +1.45% |
| 2025 | +2.44% |

The quintile test supports the existence of a volatility-related
cross-sectional pattern in this sample while again showing that the effect is
not stable across all periods.

---

## Market-Regime Diagnostics

A Gaussian Mixture Model was fitted using EM to identify latent market
environments.

The market-level inputs were:

- average 21-day stock return
- average 20-day stock volatility
- cross-sectional 21-day return dispersion

The GMM inferred three clusters, which were ordered by their fitted volatility
centers and labeled:

```text
low_vol
mid_vol
high_vol
```

Ridge Rank IC differed materially across these inferred regimes.

Overall diagnostic results were approximately:

| Regime | Mean Rank IC |
|---|---:|
| Low volatility | -0.017 |
| Mid volatility | +0.043 |
| High volatility | +0.047 |

This suggests that market conditions may help explain variation in predictive
performance.

However, regime labels were produced by an unsupervised model and should not
be interpreted as economically definitive states such as "bull" or "bear"
markets.

---

## Regime-Aware Prediction

A second experiment explicitly inserted regime information into Ridge.

The model included both regime indicators and feature-regime interactions,
allowing feature coefficients to vary across regimes.

The experiment did **not** improve prediction.

Overall Mean Rank IC fell from approximately:

```text
Baseline Ridge:      0.0256
Regime-aware Ridge:  0.0001
```

Thus, market regimes appeared useful as a diagnostic explanation of model
instability, but hard regime conditioning did not provide predictive
improvement.

---

## Raw Return vs Rank Target

The original model predicts the raw future 21-day return.

A second experiment instead trained Ridge on each stock's cross-sectional
future-return percentile rank.

This aligns the training target more directly with Rank IC, but discards
information about return magnitude.

Results:

| Target | Mean Rank IC |
|---|---:|
| Raw future return | **0.0256** |
| Future-return rank | 0.0130 |

The raw-return target performed better and was retained.

---

## Portfolio Construction

The final volatility-only Ridge predictions are converted into a portfolio.

At each rebalance date:

1. predict future returns for the stock universe;
2. rank stocks by predicted return;
3. buy the top 10%;
4. equal-weight selected names;
5. enter on the next trading day;
6. hold for approximately 21 trading days;
7. rebalance and apply turnover-based transaction costs.

The final portfolio is 100% long.

A separate long-short experiment used:

```text
+50% long
-50% short
```

but the short leg contributed little average return and mainly reduced overall
market exposure and risk.

---

## Long-Short Diagnostic

The long-short version produced approximately:

```text
CAGR:              12.65%
Annual Volatility: 15.12%
Sharpe:             0.864
Maximum Drawdown: -11.87%
```

Average contribution per holding period:

```text
Long leg:   +1.19%
Short leg:  -0.07%
```

The predictive information therefore came primarily from identifying stronger
stocks rather than identifying stocks with negative absolute future returns.

---

## Long-Only Portfolio Performance

The final long-only strategy uses the top 10% of Ridge predictions.

| Metric | Ridge Top-10% | Equal-Weight Benchmark |
|---|---:|---:|
| CAGR | **26.20%** | 9.96% |
| Annual volatility | 31.98% | 14.33% |
| Sharpe ratio | **0.885** | 0.734 |
| Maximum drawdown | -23.81% | -12.63% |
| Average turnover | 0.613 | 0.022 |
| Total return | **148.77%** | 45.04% |

![Cumulative Net Value](figures/equity_curve.png)

The Ridge-selected portfolio substantially outperformed the equal-weight
universe in raw return, but it also carried substantially higher volatility
and drawdown.

---

## Transaction-Cost Sensitivity

Turnover-based transaction costs were applied at each rebalance.

```math
\mathrm{Cost}
=
\mathrm{Turnover}
\times
\frac{\mathrm{bps}}{10000}
```

Results:

| Cost Assumption | CAGR | Sharpe |
|---|---:|---:|
| 0 bps | 26.66% | 0.90 |
| 5 bps | 26.20% | 0.88 |
| 10 bps | 25.74% | 0.87 |
| 20 bps | 24.83% | 0.85 |

The backtest does not disappear under moderately higher simplified
transaction-cost assumptions.

---

## Risk-Adjusted Benchmark Comparison

The long-only Ridge portfolio is strongly exposed to the same underlying
market movements as the equal-weight universe.

A simple single-benchmark regression is:

```math
r_{\mathrm{Ridge},t}
=
\alpha
+
\beta r_{\mathrm{Benchmark},t}
+
\varepsilon_t
```

Estimated results:

| Statistic | Value |
|---|---:|
| Beta vs benchmark | 1.921 |
| Annualized regression alpha | 8.37% |
| Correlation | 0.861 |
| $R^2$ | 0.742 |
| Information Ratio | 0.848 |
| Positive excess-return ratio | 51.06% |

The beta of approximately 1.9 shows that a substantial portion of the
portfolio's higher return is associated with greater exposure to
high-volatility / high-market-sensitivity stocks.

The positive regression alpha suggests additional stock-selection performance
relative to this simple equal-weight benchmark.

However, it should **not** be interpreted as fully factor-adjusted or
deployable trading alpha.

---

## Negative Results

Several seemingly reasonable extensions did not improve out-of-sample
performance:

- Histogram Gradient Boosting instead of Ridge
- expanding the feature set from 4 to 12 variables
- using future-return rank instead of raw future return as the target
- directly inserting hard GMM regime labels into the prediction model
- adding trend features to the volatility specification

These negative results are intentionally retained.

A major goal of the project is to distinguish real incremental information
from additional model complexity rather than reporting only successful
experiments.

---

## Key Findings

1. **More complex models were not automatically better.**  
   Ridge consistently outperformed the tested nonlinear gradient-boosting
   specification.

2. **More features were not automatically better.**  
   The volatility-only model outperformed the broader 12-feature model.

3. **Volatility contained the strongest tested cross-sectional signal.**  
   Medium- and longer-horizon volatility variables were particularly useful.

4. **The signal was time-varying.**  
   It failed in 2022 but produced positive Rank IC in 2023-2025.

5. **Market regimes helped explain instability but did not improve prediction
   when directly inserted into the model.**

6. **The signal translated into portfolio-level performance.**  
   Ridge-selected stocks substantially outperformed the equal-weight universe
   in the tested period.

7. **Much of the portfolio's risk came from high market sensitivity.**  
   The long-only portfolio had a benchmark beta of approximately 1.9.

8. **The short side was not a strong source of alpha.**  
   Most of the long-short strategy's returns came from the long leg.

---

## Limitations

This is exploratory research, not evidence of a production-ready trading
strategy.

Important limitations include:

- current constituents are projected backward, creating survivorship and
  universe-selection bias;
- the 2022-2025 period was repeatedly examined during model development and is
  therefore not a fully untouched final holdout;
- there is no point-in-time constituent membership database;
- transaction costs are simplified;
- turnover is based on target-to-target weights rather than fully
  drift-adjusted pre-rebalance weights;
- bid-ask spreads are not modeled explicitly;
- market impact and liquidity constraints are omitted;
- short borrow fees and short-sale constraints are omitted from the
  long-short diagnostic;
- the available feature set is restricted largely to historical price
  information;
- benchmark risk adjustment uses only a simple equal-weight universe rather
  than a full multi-factor model;
- overlapping 21-day future-return targets mean adjacent daily observations
  are not statistically independent;
- repeated exploratory analysis introduces data-snooping risk.

A stronger future version would use point-in-time universe membership and a
completely untouched later-period holdout after freezing all modeling choices.

---

## Project Structure

```text
ml-return-prediction/
|
|-- data/
|   |-- prices.csv
|   `-- model_dataset.csv
|
|-- figures/
|   |-- equity_curve.png
|   `-- yearly_rank_ic.png
|
|-- src/
|   |-- build_dataset.py
|   |-- train_ridge.py
|   |-- train_models.py
|   |-- walk_forward.py
|   |-- feature_group_test.py
|   |-- vol_coefficients.py
|   |-- vol_quintile_check.py
|   |-- regime_analysis.py
|   |-- regime_aware_model.py
|   |-- rank_target_test.py
|   |-- portfolio_backtest.py
|   |-- benchmark_compare.py
|   `-- make_figure.py
|
|-- README.md
`-- requirements.txt
```

---

## Requirements

```text
numpy
pandas
matplotlib
scikit-learn
```

---

## Running

Build the modeling dataset:

```bash
python src/build_dataset.py
```

Run the baseline model comparison:

```bash
python src/train_models.py
```

Run purged walk-forward evaluation:

```bash
python src/walk_forward.py
```

Run feature-group analysis:

```bash
python src/feature_group_test.py
```

Run volatility diagnostics:

```bash
python src/vol_coefficients.py
python src/vol_quintile_check.py
```

Run GMM regime diagnostics:

```bash
python src/regime_analysis.py
python src/regime_aware_model.py
```

Run the portfolio analysis:

```bash
python src/portfolio_backtest.py
python src/benchmark_compare.py
```

Generate final figures:

```bash
python src/make_figure.py
```

---

## Research Perspective

The main lesson from this project is not that one particular machine-learning
model can reliably predict stock returns.

Instead, the experiments show that weak financial signals depend strongly on:

- feature representation;
- leakage-resistant evaluation;
- market conditions;
- model complexity;
- portfolio construction;
- transaction costs;
- and underlying risk exposure.

In this experiment, simple models combined with careful out-of-sample
diagnostics were more informative than increasing model complexity.
