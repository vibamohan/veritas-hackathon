# Model experiments: architectures, horizons, and validation periods

Code: [`scripts/experiments.py`](../scripts/experiments.py). Original comparison: 2026-11-14; additional regression comparison: 2026-11-15.

## Question

Can any model beat the simplest baseline, "always predict the training-period average" (for raw returns, always up), at predicting the direction of NVDA, TSLA, AAPL and SPY? We tested 3 horizons, 2 targets and 13 models, on 7 validation years plus one untouched test period.

## Method

- **Horizons:** 1, 5 and 20 trading days.
- **Targets:**
  - **return:** the stock's own forward return. Includes SPY.
  - **excess:** the stock's return minus SPY's over the same window. Asks "will it beat the market?", which removes the market's upward drift. NVDA, TSLA and AAPL only.
- **Walk-forward validation, 2015–2021:** for each year, train on all earlier data (from 2010), with the last *h* training rows purged so no label crosses into the scored year. Score that year with non-overlapping forecasts, one every *h* days. This gives 7 regimes per ticker, including the 2018 drawdown and the 2020 crash.
- **Test:** January 2022 to September 2023 (the historical news datasets end around October 2023). The test was run once, for one model chosen in advance by the rule below.
- **Point-in-time news:**
  - The query for day *t* is the 10 newest headlines dated in the 7 days strictly before *t*.
  - Matched past days must be older than the query window, and their *h*-day outcome must have finished by *t*.
  - The TF-IDF vocabulary is fit on pre-2015 headlines only. Embeddings come from a pretrained model (`all-MiniLM-L6-v2`), so no fitting is involved.
  - GDELT is live-only and is excluded.
- **Metrics:**
  - **Edge:** directional hit rate minus the baseline's, in percentage points. One standard error is about 0.6, 1.3 and 2.5 points for *h* = 1, 5 and 20.
  - **IC:** Spearman rank correlation between forecast and outcome, averaged over ticker-years. IC t is that mean divided by its standard error.
- **Selection rule, fixed before any test run:** take the best validation result. Because nothing beat the baseline on hit rate, the tiebreak was the highest IC t-stat.

## Models

| Model | What it is |
| --- | --- |
| Always up (train mean) | Baseline: predict the training-period mean |
| Ridge base (current) | The app's model: StandardScaler + Ridge(α=1) on 6 features (1d return, 5/20d momentum, 20d volatility, volume ratio, SPY 5d) |
| Ridge ext | Ridge with cross-validated α (0.01–100) on 11 features: the 6 above plus 60d momentum, 60d volatility, distance from the 50-day and 200-day moving averages, and SPY 20d |
| Logistic ext | Logistic regression on direction, 11 features |
| HGB ext / HGB classifier ext | Gradient-boosted trees (depth 3, 200 rounds, regularised), regression or classification |
| News kNN TF-IDF | The dashboard's news model: similarity-weighted outcome of the 15 most similar past headline days |
| News kNN embeddings | Same, with sentence embeddings instead of TF-IDF |
| Ridge / HGB ext + news | 11 features + log headline count + both kNN news forecasts as inputs |
| …, gated | Says "down" only when the model's forecast falls in its most bearish 20% of training-period forecasts |

## Validation results (2015–2021)

Each cell is edge (points) / mean IC. ✱ marks IC t ≥ 2.

**Target: return** (n = 7,052 / 1,428 / 364; baseline hit rate 53.9% / 60.4% / 66.5%)

| Model | 1 day | 5 days | 20 days |
| --- | ---: | ---: | ---: |
| Ridge base (current) | −1.5 / +0.016 | −3.1 / −0.015 | −2.2 / −0.035 |
| Ridge ext | −1.9 / +0.018 | −5.5 / +0.008 | −6.6 / −0.060 |
| Logistic ext | −1.5 / +0.015 | −4.1 / +0.013 | −5.8 / −0.056 |
| HGB ext | −3.0 / +0.025 | −8.3 / +0.017 | −10.2 / −0.024 |
| HGB classifier ext | −2.6 / +0.016 | −9.7 / +0.013 | −12.1 / +0.009 |
| News kNN TF-IDF | −1.8 / +0.022 | −4.7 / +0.006 | −4.7 / −0.003 |
| News kNN embeddings | −1.6 / +0.016 | −4.1 / −0.024 | −4.4 / −0.040 |
| Ridge ext + news | −1.8 / +0.020 | −5.7 / +0.003 | −7.1 / −0.060 |
| HGB ext + news | −3.5 / +0.021 | −10.2 / −0.008 | −9.1 / +0.033 |
| Ridge ext + news, gated | −1.3 / +0.020 | −6.6 / +0.003 | −11.0 / −0.060 |
| Logistic ext, gated | −2.6 / +0.015 | −5.8 / +0.013 | −8.8 / −0.056 |
| News kNN embeddings, gated | −1.2 / +0.016 | −2.9 / −0.024 | −4.4 / −0.040 |

**Target: excess over SPY** (n = 5,289 / 1,071 / 273; "always outperform" hit rate 51.9% / 55.4% / 59.3%)

| Model | 1 day | 5 days | 20 days |
| --- | ---: | ---: | ---: |
| Always up (train mean) | −0.6 / — | −0.5 / — | −0.4 / — |
| Ridge base (current) | −0.2 / +0.019 | −0.3 / −0.039 | −2.9 / −0.062 |
| Ridge ext | −1.6 / +0.003 | −4.9 / −0.056 ✱ | −8.8 / −0.078 |
| Logistic ext | −2.3 / +0.005 | −4.0 / −0.017 | −8.4 / −0.066 |
| HGB ext | −0.5 / +0.038 ✱ | −4.4 / +0.002 | −8.4 / +0.031 |
| HGB classifier ext | −0.4 / +0.028 ✱ | −4.0 / −0.004 | −5.9 / +0.050 |
| **News kNN TF-IDF** | **−0.2 / +0.036 ✱** | −3.7 / +0.002 | −2.2 / −0.000 |
| News kNN embeddings | −1.2 / +0.002 | −3.3 / −0.012 | −1.1 / +0.029 |
| Ridge ext + news | −1.2 / +0.007 | −3.8 / −0.057 ✱ | −9.9 / −0.098 |
| HGB ext + news | −0.5 / +0.028 ✱ | −4.9 / −0.019 | −8.4 / +0.041 |
| Ridge ext + news, gated | −0.2 / +0.007 | −4.8 / −0.057 ✱ | −7.0 / −0.098 |
| Logistic ext, gated | −1.6 / +0.005 | −3.2 / −0.017 | −7.0 / −0.066 |
| News kNN embeddings, gated | −0.1 / +0.002 | +0.0 / −0.012 | +0.0 / +0.029 |

(The ✱ on Ridge models at 5 days marks a significantly *negative* IC.)

## Test result (Jan 2022 – Sep 2023, run once)

Selected model: **News kNN TF-IDF, 1-day, excess-over-SPY target**, the highest validation IC t-stat (2.6).

| Model | Hit rate | Edge vs always-outperform | Mean IC | NVDA / TSLA / AAPL hits (of 438) |
| --- | ---: | ---: | ---: | --- |
| Always outperform | 51.8% | — | — | 221 / 227 / 233 |
| Ridge base (current) | 49.8% | −2.1 | +0.017 | 224 / 220 / 210 |
| **News kNN TF-IDF** | **53.4%** | **+1.6 (SE 1.4)** | **+0.034** | **232 / 235 / 235** |

## Conclusions

1. **Raw-return direction: nothing beats always-up** at any horizon. These seven years mostly went up, so a model that ever says "down" pays for it. More features, trees, classifiers, embeddings and gating all made this worse, not better. The longer the horizon, the stronger the drift, and the worse the models do against it.
2. **The current dashboard setup (5-day, 70/30 price + news blend) has no support in the evidence.** At 5 days, both the price model and the news model lose to the baseline in both targets.
3. **The only signal that held up is short-horizon and relative.** The news-similarity model (TF-IDF) predicting next-day return *relative to SPY*:
   - had a positive IC in validation;
   - beat the baseline on held-out 2022–2023 data for each of the three stocks, with an IC (+0.034) matching validation.
4. **That signal is weak.** +1.6 points is about 1.1 standard errors. Across the 78 validation combinations, a t-stat around 2.5 can appear by chance. This is consistent with a small real effect but does not prove one, and it is not evidence of a tradable strategy: no transaction costs were modelled, and the test window is 21 months.
5. **Sentence embeddings did not beat TF-IDF here.** Headlines are short and ticker-specific, so exact word overlap (company names, products, "earnings", "recall") seems to carry the useful information.

## Additional regression comparison (2026-11-15)

Added ElasticNet, Random Forest, and Extra Trees regressors and reran the same
2015–2021 walk-forward folds for both targets. ElasticNet's alpha and mixing
weight are selected with expanding time-series folds and a 20-session gap. The table gives directional edge
over the period-specific always-up baseline (percentage points) and mean
year/ticker Spearman IC. None of these validation scores is a fresh holdout.

| Target | Model | 1 day (edge / IC) | 5 days (edge / IC) | 20 days (edge / IC) |
| --- | --- | ---: | ---: | ---: |
| Return | ElasticNet | −0.2 / +0.099 | −0.6 / +0.063 | −1.4 / +0.140 |
| Return | Random Forest | −3.5 / +0.034 | −8.6 / +0.018 | −12.6 / +0.013 |
| Return | Extra Trees | −1.6 / +0.026 | −5.3 / −0.004 | −7.1 / −0.019 |
| Excess over SPY | ElasticNet | −0.7 / +0.007 | −0.2 / −0.062 | −0.4 / +0.023 |
| Excess over SPY | Random Forest | −0.4 / +0.039 | −3.9 / +0.004 | −10.3 / +0.027 |
| Excess over SPY | Extra Trees | +0.8 / +0.029 | −2.3 / −0.008 | −8.1 / −0.010 |

The Extra Trees one-day excess-return hit-rate edge is +0.8 points with an
estimated standard error of 0.7 points. This is small, not a decisive win, and
comes from a broad model search. The stronger one-day rank correlations for
Random Forest and Extra Trees did not translate into better hit rates. At five
and twenty days, all added models were at or below the baseline. Price refreshes
were unavailable during this run, so existing cached price files were used.
ElasticNet had positive mean rank IC for raw returns, but its directional hit
rate remained below the baseline at all three horizons. That IC did not yield
useful sign decisions in this evaluation.

LLM scoring is now an optional feature path in `scripts/llm_news.py`; it was not
run for this comparison because no `OPENAI_API_KEY` is configured. When scores
are generated, `scripts/experiments.py` adds point-in-time headline sentiment
and confidence features and evaluates Ridge, ElasticNet, and HGB with those
features. Those model rows are skipped when no cached LLM scores exist.

## Recommendations for the app

- Present the forecast honestly. Show the 1-day "vs SPY" news signal with its measured hit rate (53% against a 52% baseline), or show the matched historical headlines as context only. Drop the 5-day 70/30 blend.
- The 2022–2023 test window has now been used. Any further tuning should be validated on 2022–2023 and then tested on newer data (for example, collect GDELT headlines going forward).
- Ideas not tried yet:
  - a finance-specific sentiment model (such as FinBERT) as a feature;
  - event-type tags from the Fliar BI data (M&A, lawsuit, layoffs);
  - pooling all tickers into one model to get more training data.

## Reproduce

```bash
uv run --with sentence-transformers python scripts/experiments.py                  # validation, return target
uv run --with sentence-transformers python scripts/experiments.py --target excess  # validation, excess target
uv run python scripts/experiments.py --test "News kNN TF-IDF" 1 --target excess    # the one test run
uv run python scripts/llm_news.py --ticker NVDA                                   # optional LLM headline scores
```

Without `--with sentence-transformers` the embedding models are skipped. Raw per-ticker, per-year results are written to `data/cache/experiments_validation_<target>.csv`. A full validation run takes about 4–5 minutes on a laptop.
