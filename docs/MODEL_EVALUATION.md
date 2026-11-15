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

## News ablation (point-in-time)

`uv run python scripts/evaluate.py [TICKER ...]` scores the news half on the same validation dates (2020–2021, every 5 trading days, ~99 forecasts per ticker). For each forecast date *t*, the query is the 10 newest historical headlines from the 7 days strictly before *t*. The candidate matches are past days whose 5-day outcome had finished by *t*. The TF-IDF vocabulary is fit on pre-2020 headlines only. GDELT is live-only and is excluded. When there is no recent news, the news forecast falls back to the price model, as the dashboard does.

| Ticker | Model | MAE | Direction | Corr | News coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| NVDA | Always up (+1%) | 4.9% | 62.6% | — | 97/99 |
| NVDA | Price model | 4.9% | 62.6% | -0.14 | |
| NVDA | News only | 5.3% | 50.5% | -0.26 | |
| NVDA | 70/30 blend | 5.0% | 56.6% | -0.31 | |
| TSLA | Always up (+1%) | 7.4% | 60.6% | — | 99/99 |
| TSLA | Price model | 7.4% | 60.6% | -0.12 | |
| TSLA | News only | 8.0% | 52.5% | -0.07 | |
| TSLA | 70/30 blend | 7.6% | 58.6% | -0.12 | |
| AAPL | Always up (+1%) | 3.3% | 63.6% | — | 99/99 |
| AAPL | Price model | 3.4% | 61.6% | -0.10 | |
| AAPL | News only | 3.4% | 64.6% | 0.10 | |
| AAPL | 70/30 blend | 3.4% | 59.6% | 0.01 | |
| SPY | Always up (+1%) | 2.0% | 65.7% | — | 45/99 |
| SPY | Price model | 2.1% | 58.6% | -0.12 | |
| SPY | News only | 2.1% | 60.6% | 0.21 | |
| SPY | 70/30 blend | 2.1% | 60.6% | 0.05 | |

**Reading:** no model beats always-up on all four tickers. On NVDA and TSLA, adding news makes the forecast worse. AAPL news-only is +1 hit out of 99, and SPY news-only shows a weak positive correlation on 45 covered dates. Both are within noise: with n≈99, one standard error of directional accuracy is about ±5 points. 2020–2021 was a strong bull market, which favors always-up. The 30% news weight on the dashboard is not supported by validation evidence.

## Tuning rules

Use the 2020–2021 validation results to choose features, model family, and hyperparameters. Keep 2022 onward untouched until the approach is selected. Any change made after looking at test scores means that test period has become development data; use a newer holdout for a fresh final estimate. Report all tickers and multiple periods, retain simple baselines, and prefer a model only when it improves validation without fragile ticker-specific gains.

Next experiments should be small and measured: Ridge alpha sweep; ElasticNet; tree boosting/ExtraTrees. The leakage-safe news ablation is done (above); next news ideas are sentence embeddings instead of TF-IDF and news features (headline count, top similarity) fed to the model. News replay must use only headlines available by each forecast date, and historical neighbor outcomes must have completed before that date. Publication-date quality differs across source datasets. LLM event extraction is a later experiment only if dated headline features show incremental validation value.

## Further experiments

See [EXPERIMENTS.md](EXPERIMENTS.md) for the 13-model comparison: 1/5/20-day horizons, raw and excess-over-SPY targets, walk-forward validation over 2015–2021, and a single 2022–2023 test run.

See [VOLATILITY.md](VOLATILITY.md) for the volatility model ("will the coming week be turbulent?"). It reaches 76.7% accuracy on held-out 2022–2023 data at 5 days, against 66.5% for the best baseline.
