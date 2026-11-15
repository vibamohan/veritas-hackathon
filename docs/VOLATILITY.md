# Volatility forecasting: "will the coming week be turbulent?"

Code: [`scripts/volatility.py`](../scripts/volatility.py). Run date: 2026-11-15.

## Why volatility

[EXPERIMENTS.md](EXPERIMENTS.md) showed that no model beats "always up" at predicting price *direction*, and that an honest direction edge is about 1–2 points. Volatility, meaning how *much* a stock moves regardless of direction, is a different story. It clusters, it rises around scheduled events, and the options market prices it (VIX). Predicting it is useful (risk, position sizing, "brace for a big week"), and it can be tested honestly against strong baselines.

## Target

For horizon *h* ∈ {1, 5, 20} trading days:

- **turbulent** = realised volatility over the next *h* days, √mean(r²), is **above the stock's trailing 1-year median** of *h*-day volatility;
- **calm** otherwise.

The threshold uses only past data, so roughly half of all periods are turbulent.

## Baselines

- **Majority class:** always predict whichever label was more common in training.
- **Persistence:** "the next *h* days will be like the last *h* days". Volatility clusters, so this is the strong baseline to beat.

## Features (all known at the close of day *t*)

| Group | Features |
| --- | --- |
| Recent volatility | 1, 5, 20 and 60-day realised volatility relative to the 1-year median; SPY 20-day volatility relative to its median |
| Market fear | VIX relative to its 1-year median; VIX 5-day change |
| Scheduled events | Earnings reaction day within the next *h* days; days to next / since last earnings. Earnings dates come from yfinance; after-close reports count on the next session. Dates are announced weeks ahead, so this is not leakage. |
| Trend | 1-day return, 20-day momentum, distance from the 200-day moving average |
| News | Log headline count in the prior 7 days; the volatility that followed the 15 most similar past headline days (TF-IDF kNN, point-in-time as in EXPERIMENTS.md) |

## Models

| Model | Description |
| --- | --- |
| VIX-only logistic | Logistic regression on VIX, SPY volatility and the stock's 20-day volatility only |
| Logistic | Logistic regression (C = 0.1) on all features, one per ticker |
| HGB / HGB, no news | Gradient-boosted trees, with and without the news features |
| Random forest / Extra trees | 300 trees, minimum leaf size 30 |
| Ensemble | Average probability of logistic + HGB + random forest |
| Pooled … | The same models trained once on all four tickers' rows: 4× the data, and features that are already relative to each stock's own normal |

## Method

- **Validation:** walk-forward, one fold per year 2015–2021. Each fold trains on 2011 to the end of the previous year, with the last *h* rows purged, and scores non-overlapping forecasts.
- **Test:** January 2022 to September 2023, run once.
- **Selection rule, fixed in advance:** the best validation accuracy picks the model. Pooled logistic was best at every horizon, so it is the only model tested.

## Validation results (2015–2021)

Accuracy (AUC in brackets). n = 7,052 / 1,428 / 364 forecasts for *h* = 1 / 5 / 20. Standard error ≈ 0.6 / 1.2 / 2.4 points.

| Model | 1 day | 5 days | 20 days |
| --- | ---: | ---: | ---: |
| Majority class | 45.9% | 44.2% | 46.2% |
| Persistence | 53.9% | 62.4% | 64.6% |
| VIX-only logistic | 59.0% (0.63) | 65.3% (0.72) | 64.6% (0.72) |
| Logistic | 59.9% (0.64) | 67.4% (0.74) | 71.4% (0.78) |
| HGB | 58.8% (0.62) | 63.8% (0.70) | 67.3% (0.74) |
| HGB, no news | 57.8% (0.61) | 64.1% (0.70) | 68.7% (0.76) |
| Random forest | 59.7% (0.63) | 66.6% (0.73) | 69.8% (0.77) |
| Extra trees | 60.2% (0.64) | 66.5% (0.74) | 71.4% (0.78) |
| Ensemble | 59.8% (0.64) | 66.5% (0.74) | 71.4% (0.78) |
| **Pooled logistic** | **60.7% (0.65)** | **68.0% (0.74)** | **73.9% (0.79)** |
| Pooled HGB | 60.0% (0.64) | 66.5% (0.74) | 69.8% (0.76) |
| Pooled random forest | 60.1% (0.65) | 66.9% (0.74) | 70.9% (0.76) |
| Pooled ensemble | 60.5% (0.65) | 66.9% (0.75) | 71.4% (0.79) |

## Test results (Jan 2022 – Sep 2023, run once)

| Horizon | Pooled logistic | AUC | Persistence | Majority | AAPL / NVDA / SPY / TSLA | n |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 1 day | **61.0%** | 0.64 | 50.7% | 51.2% | 63.5 / 59.8 / 64.6 / 56.2% | 1,752 |
| **5 days** | **76.7%** | **0.86** | 66.5% | 48.6% | 77.3 / 77.3 / 83.0 / 69.3% | 352 |
| 20 days | 79.5% | 0.86 | **80.7%** | 53.4% | 77.3 / 72.7 / 90.9 / 77.3% | 88 |

## What drives it

Standardised coefficients of the 5-day pooled logistic model, trained on all pre-2022 data:

| Feature | Coefficient | Meaning |
| --- | ---: | --- |
| VIX vs its 1-year median | +0.65 | Market fear spills into every stock |
| Earnings in the next 5 days | +0.42 | Scheduled event: earnings days are turbulent 86% of the time for NVDA |
| Stock's 20-day volatility | +0.40 | Volatility clusters |
| SPY's 20-day volatility | −0.26 | Given VIX, a stock-specific shock fades faster than a market-wide one |
| 5 and 60-day volatility | +0.20 / +0.19 | |
| 20-day momentum | −0.19 | Sell-offs are more turbulent than rallies |
| News kNN, headline count | −0.10 / −0.06 | Small. News adds little on top of VIX and earnings |

## Conclusions

1. **The headline result is the 5-day turbulence forecast: 76.7% accuracy on held-out 2022–2023 data** (68.0% in validation), 10 points above persistence and 28 above the majority class, with AUC 0.86. It beats persistence on every ticker. Test accuracy is higher than validation, probably because 2022's bear market had long, clear volatility regimes. Expect roughly 68–77% going forward.
2. **At 20 days**, the 74–80% accuracy is real but no better than persistence on the test set (n = 88, so ±4 points). Monthly volatility regimes are slow-moving, so "same as last month" is already very good.
3. **At 1 day**, about 61% is a clear edge over the 51–54% baselines, but daily turbulence is noisy.
4. **Pooling all tickers into one model helped most**, more than the choice of algorithm. A simple, regularised logistic regression beat the trees. The useful information is a few strong, roughly linear signals: VIX, scheduled earnings, and recent volatility.
5. **News features barely matter for volatility.** HGB with and without news is within noise, and the news coefficients are small. The news-similarity idea remains useful as explanation ("past weeks with headlines like this"), not as the main driver.

## Caveats

- One test window (21 months), and an unusual one (2022 bear market). Accuracy in calm years will likely be closer to the validation 68%.
- "Turbulent" means above the stock's own 1-year median, not an absolute level.
- VIX is a market-implied forecast. Using it is legitimate (it is known at the close), but part of the skill is borrowed from the options market.
- The earnings calendar comes from yfinance's last 100 reports (back to 2002; TSLA from 2010), checked by eye, not against an official source.

## Reproduce

```bash
uv run python scripts/volatility.py                          # validation, 11 models × 3 horizons (~10 min)
uv run python scripts/volatility.py --test "Pooled logistic" 5  # the test run (also run for 1 and 20)
```

Per-ticker, per-year validation results are written to `data/cache/volatility_validation.csv`.
