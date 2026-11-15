# Stock model architecture and evaluation

## Current model

The five-trading-day return forecast uses one-day return, five-day and twenty-day momentum, twenty-day annualized volatility, volume divided by its twenty-day average, and five-day SPY return. A `StandardScaler` normalizes those six features before `Ridge(alpha=1)` predicts the next-five-session return.

The dashboard also blends in a TF-IDF/cosine-similarity news-neighbor estimate at 30%. The validation comparison below evaluates the market Ridge model only. News should be enabled in a forecast only after a point-in-time historical replay verifies that it adds value.

## Fixed chronological evaluation

`GET /api/replay?ticker=NVDA` follows this split:

- **Train:** through 2019-12-31.
- **Validation:** 2020-01-01 through 2021-12-31.
- **Test:** 2022-01-01 onward. The endpoint returns test forecasts separately; do not tune on them.
- **Purging:** remove the final five trading rows before each boundary and the first five rows after it. Each target is the following five-session return, so this prevents a label window from crossing a split boundary.
- **Sampling:** validation and test forecasts are spaced five trading sessions apart to avoid overlapping scored outcomes.

The fixed comparison scores the current Ridge model against an always-up baseline that forecasts +1% over five sessions. Both receive MAE, RMSE, directional accuracy, mean predicted/actual return, correlation, and sample count. The baseline's return magnitude is only a scoring convention; its directional prediction is always positive.

## Interpreting the first cached results

| Ticker | Model | MAE | Direction accuracy | N |
| --- | --- | ---: | ---: | ---: |
| NVDA | Ridge | 4.93% | 62.6% | 99 |
| NVDA | Always up (+1%) | 4.87% | 62.6% | 99 |
| SPY | Ridge | 2.10% | 58.6% | 99 |
| SPY | Always up (+1%) | 2.03% | 65.7% | 99 |

On these cached 2020–2021 validation observations, the current model does not beat always-up: NVDA has the same direction rate and slightly worse MAE; SPY is worse on both. The validation set is small and a single regime, so this is evidence about this evaluation slice, not general predictive ability.

## Tuning rules

Use the 2020–2021 validation results to choose features, model family, and hyperparameters. Keep 2022 onward untouched until the approach is selected. Any change made after looking at test scores means that test period has become development data; use a newer holdout for a fresh final estimate. Report all tickers and multiple periods, retain simple baselines, and prefer a model only when it improves validation without fragile ticker-specific gains.

Next experiments should be small and measured: Ridge alpha sweep; ElasticNet; tree boosting/ExtraTrees; then a leakage-safe news ablation. News replay must use only headlines available by each forecast date, and historical neighbor outcomes must have completed before that date. Publication-date quality differs across source datasets. LLM event extraction is a later experiment only if dated headline features show incremental validation value.
