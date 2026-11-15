"""Volatility model: will the next h trading days be more turbulent than usual? (results: docs/VOLATILITY.md)

Target ("turbulent"): the stock's realised volatility over the next h days, sqrt(mean(daily return^2)),
is above its own trailing 1-year median. Direction doesn't matter, only the size of the moves.
Baselines: "majority class" (always guess the more common label) and "persistence"
(guess that the next h days will be like the last h days).

Validation (2015–2021, one fold per year) and the single test run (2022-01 to 2023-09) work as in
experiments.py. Every feature is known at the close of day t: earnings dates are announced weeks in
advance, and news comes from the 7 days before t.

The dashboard calls outlook() at the bottom of this file.

Usage:
  uv run python scripts/volatility.py                                 # validation, all models (~12 min)
  uv run python scripts/volatility.py --test "Pooled logistic" 5      # one test run on 2022-2023
"""
import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import ElasticNetCV, LogisticRegression, Ridge, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for echomarket
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for experiments (also when app.py imports this)
from echomarket import CACHE, TICKERS, prices
from experiments import TEST, VAL_YEARS, news_features

HORIZONS = (1, 5, 20)
TRADING_DAYS_PER_YEAR = 252


# ===========================================================================
# Data
# ===========================================================================

class TickerData(NamedTuple):
    X: pd.DataFrame  # features, one row per trading day
    turbulent: pd.DataFrame  # target per horizon: True / False (NaN when the future isn't known yet)
    persistence: pd.DataFrame  # baseline guess per horizon: were the last h days turbulent?
    log_vol_ratio: pd.DataFrame  # continuous target per horizon: log(future vol / normal vol)


def realised_vol(returns, window):
    """Root-mean-square daily return over a trailing window (a standard volatility measure)."""
    return np.sqrt((returns ** 2).rolling(window).mean())


def earnings_reaction_positions(ticker, index):
    """Row numbers (in `index`) of the trading day each earnings report first moves the stock.
    Upcoming reports after the last row get row numbers past the end, counted in business days.
    SPY is an index fund and has no earnings."""
    if ticker == "SPY":
        return np.array([], dtype=int)
    path = CACHE / f"earnings_{ticker}.parquet"
    if path.exists() and time.time() - path.stat().st_mtime < 86400:  # refresh daily for new dates
        stamps = pd.read_parquet(path).date
    else:
        reports = yf.Ticker(ticker).get_earnings_dates(limit=100)
        stamps = pd.Series(reports.index.tz_convert("America/New_York").tz_localize(None), name="date")
        stamps.to_frame().to_parquet(path)
    stamps = pd.to_datetime(stamps)

    # Reports released after the 16:00 close move the stock on the next session.
    after_close = (stamps.dt.hour >= 16).astype(int)
    reaction = pd.DatetimeIndex(stamps.dt.normalize() + pd.to_timedelta(after_close, unit="D")).as_unit(index.unit)
    positions = index.searchsorted(reaction)  # first trading day on or after the reaction date
    future = positions >= len(index)
    last_day = np.datetime64(index[-1].date(), "D")
    positions[future] = len(index) - 1 + np.busday_count(last_day, reaction[future].to_numpy().astype("datetime64[D]"))
    return np.unique(positions)


def earnings_timing(ticker, index):
    """For every row: trading days until the next earnings reaction, and since the last one."""
    rows = np.arange(len(index))
    reactions = np.append(earnings_reaction_positions(ticker, index), 10 ** 9)  # sentinel = "no next report"
    next_i = np.searchsorted(reactions, rows, side="right")  # first reaction strictly after each row
    days_to = reactions[next_i] - rows
    previous = reactions[np.maximum(next_i - 1, 0)]
    days_since = np.where((next_i > 0) & (previous <= rows), rows - previous, 999)
    return days_to, days_since


def frame(ticker, news=True):
    """Features and targets for one ticker. news=False skips the (slower) news features."""
    stock, spy, vix = prices(ticker), prices("SPY"), prices("^VIX")
    close = stock["Close"]
    returns = close.pct_change()
    spy_returns = spy["Close"].reindex(stock.index).ffill().pct_change()
    vix_level = vix["Close"].reindex(stock.index).ffill()
    days_to, days_since = earnings_timing(ticker, stock.index)

    X = pd.DataFrame(index=stock.index)
    turbulent, persistence, log_vol_ratio = {}, {}, {}
    for h in HORIZONS:
        past = realised_vol(returns, h)  # volatility over the last h days
        normal = past.rolling(TRADING_DAYS_PER_YEAR).median()  # this stock's usual level (past year)
        future = past.shift(-h)  # volatility over the next h days
        turbulent[h] = (future > normal).where(future.notna())
        log_vol_ratio[h] = np.log((future + 1e-6) / (normal + 1e-6))  # +1e-6: a flat day has vol 0
        persistence[h] = past > normal
        X[f"earn_in_{h}"] = ((days_to >= 1) & (days_to <= h)).astype(float)  # earnings within the window?

    # Volatility features, all relative to "normal" so different stocks are comparable.
    normal20 = realised_vol(returns, 20).rolling(TRADING_DAYS_PER_YEAR).median()
    for window in (1, 5, 20, 60):
        X[f"rv{window}"] = np.log(realised_vol(returns, window) / normal20 + 1e-6)
    spy_vol = realised_vol(spy_returns, 20)
    X["spy_rv20"] = np.log(spy_vol / spy_vol.rolling(TRADING_DAYS_PER_YEAR).median())
    X["vix"] = np.log(vix_level / vix_level.rolling(TRADING_DAYS_PER_YEAR).median())  # market fear vs normal
    X["vix_chg5"] = np.log(vix_level / vix_level.shift(5))
    X["days_to_earn"] = np.minimum(days_to, 70)
    X["days_since_earn"] = np.minimum(days_since, 70)
    X["ret1"] = returns
    X["mom20"] = close.pct_change(20)
    X["ma200_gap"] = close / close.rolling(200).mean() - 1

    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    X = X[X.index >= "2011-01-01"]
    data = TickerData(
        X,
        pd.DataFrame(turbulent).reindex(X.index),
        pd.DataFrame(persistence).reindex(X.index),
        pd.DataFrame(log_vol_ratio).reindex(X.index),
    )
    if not news:
        return data

    # News: headline volume, and the volatility that followed the most similar past headline days.
    nf = news_features(ticker, X, data.log_vol_ratio)
    X["news_count"] = nf["news_count"]
    for h in HORIZONS:
        X[f"knn_vol_{h}"] = nf[("knn_tfidf", h)]
    return data._replace(X=X.replace([np.inf, -np.inf], np.nan))


# ===========================================================================
# Models
# ===========================================================================

@dataclass
class Model:
    estimator: str  # see make_estimator()
    features: str = "all"  # "all", "no_news", "vix", "har" or "har+"; see feature_columns()
    pooled: bool = False  # True: one model trained on all four tickers' rows


CLASSIFIERS = ("logistic", "hgb", "random_forest", "extra_trees", "ensemble")
REGRESSORS = ("ols", "ridge", "elastic_net", "hgb_regression")  # fit log_vol_ratio instead of turbulent

MODELS = {
    "VIX-only logistic": Model("logistic", features="vix"),
    "Logistic": Model("logistic"),
    "HGB": Model("hgb"),
    "HGB, no news": Model("hgb", features="no_news"),
    "Random forest": Model("random_forest"),
    "Extra trees": Model("extra_trees"),
    "Ensemble (logit+HGB+RF)": Model("ensemble"),
    "Pooled logistic": Model("logistic", pooled=True),
    "Pooled HGB": Model("hgb", pooled=True),
    "Pooled random forest": Model("random_forest", pooled=True),
    "Pooled ensemble": Model("ensemble", pooled=True),
    "Pooled logistic, no news": Model("logistic", features="no_news", pooled=True),  # the dashboard's live model
    # Regressions on the continuous log volatility ratio; "turbulent" when the prediction is above 0.
    "HAR regression": Model("ols", features="har", pooled=True),  # textbook HAR: daily/weekly/monthly vol
    "HAR + VIX + earnings": Model("ols", features="har+", pooled=True),
    "Ridge regression": Model("ridge", pooled=True),
    "ElasticNet regression": Model("elastic_net", pooled=True),
    "HGB regression": Model("hgb_regression", pooled=True),
}


def make_estimator(name):
    """A fresh, untrained scikit-learn model."""
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    if name == "hgb":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                              min_samples_leaf=50, l2_regularization=1.0, random_state=0)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=30, max_features=0.5, n_jobs=-1, random_state=0)
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=30, max_features=0.5, n_jobs=-1, random_state=0)
    if name == "ols":  # plain linear regression (a tiny ridge penalty only for numerical stability)
        return make_pipeline(StandardScaler(), Ridge(alpha=1e-6))
    if name == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 11)))
    if name == "elastic_net":
        return make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=(0.1, 0.5, 0.9),
                                                            cv=TimeSeriesSplit(n_splits=5, gap=20), max_iter=20000))
    if name == "hgb_regression":
        return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300,
                                             min_samples_leaf=50, l2_regularization=1.0, random_state=0)
    raise ValueError(f"unknown estimator {name}")


def feature_columns(X, h, features="all"):
    """Which columns a model uses at horizon h."""
    if features == "vix":
        return ["vix", "spy_rv20", "rv20"]
    if features == "har":
        return ["rv1", "rv5", "rv20"]
    if features == "har+":
        return ["rv1", "rv5", "rv20", "vix", f"earn_in_{h}"]
    # Every feature, but only this horizon's earnings flag and news forecast.
    cols = [c for c in X if not c.startswith(("earn_in_", "knn_vol_"))] + [f"earn_in_{h}"]
    if features == "no_news":
        return [c for c in cols if c != "news_count"]
    return cols + [f"knn_vol_{h}"]


def predict_turbulence(estimator, X_train, y_train, X_eval):
    """Train and return a score per row where > 0.5 means 'turbulent'.
    Classifiers return P(turbulent). Regressors predict the log volatility ratio and return
    0.5 + prediction, so the 0.5 threshold and the AUC ranking still apply."""
    if estimator == "ensemble":
        return np.mean([predict_turbulence(e, X_train, y_train, X_eval) for e in ("logistic", "hgb", "random_forest")], axis=0)
    fill = X_train.median()  # knn_vol is missing on days without recent news
    model = make_estimator(estimator).fit(X_train.fillna(fill), y_train)
    if estimator in REGRESSORS:
        return 0.5 + model.predict(X_eval.fillna(fill))
    return model.predict_proba(X_eval.fillna(fill))[:, 1]


# ===========================================================================
# Evaluation
# ===========================================================================

def periods(test):
    if test:
        return [TEST]
    return [(pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")) for y in VAL_YEARS]


def split_rows(d, h, start, end):
    """Training rows (before `start`, last h purged) and scored rows (every h-th day in the period)."""
    has_label = d.turbulent[h].notna().to_numpy()
    train = np.flatnonzero((d.X.index < start) & has_label)[:-h]
    scored = np.flatnonzero((d.X.index >= start) & (d.X.index < end) & has_label)[::h]
    return train, scored


def baseline_results(data, splits, h, year):
    rows = []
    for t, d in data.items():
        train, scored = splits[t]
        actual = d.turbulent[h].iloc[scored].astype(bool).to_numpy()
        majority = d.turbulent[h].iloc[train].mean() > 0.5
        for name, guess in [("Majority class", np.full(len(scored), majority)),
                            ("Persistence (last h days)", d.persistence[h].iloc[scored].to_numpy())]:
            rows.append(dict(model=name, h=h, ticker=t, year=year, n=len(scored),
                             hits=int((guess == actual).sum()), proba=None, y=None))
    return rows


def model_scores(spec, data, splits, h):
    """Scores for every ticker's scored rows: from one pooled model, or one model per ticker."""
    cols = feature_columns(data[TICKERS[0]].X, h, spec.features)
    regression = spec.estimator in REGRESSORS

    def training_data(d, rows):
        target = d.log_vol_ratio[h] if regression else d.turbulent[h].astype(bool)
        return d.X[cols].iloc[rows], target.iloc[rows]

    eval_frames = [data[t].X[cols].iloc[splits[t][1]] for t in TICKERS]
    if spec.pooled:  # fit once on all tickers' training rows, then split the scores back per ticker
        parts = [training_data(data[t], splits[t][0]) for t in TICKERS]
        X_train = pd.concat([p[0] for p in parts])
        y_train = pd.concat([p[1] for p in parts])
        scores = predict_turbulence(spec.estimator, X_train, y_train, pd.concat(eval_frames))
        return np.split(scores, np.cumsum([len(f) for f in eval_frames])[:-1])
    return [predict_turbulence(spec.estimator, *training_data(data[t], splits[t][0]), frame_)
            for t, frame_ in zip(TICKERS, eval_frames)]


def run(models, horizons, test=False):
    """Score the baselines and every model in `models`. One result row per ticker, horizon and period."""
    data = {t: frame(t) for t in TICKERS}
    rows = []
    for h in horizons:
        for start, end in periods(test):
            splits = {t: split_rows(d, h, start, end) for t, d in data.items()}
            rows += baseline_results(data, splits, h, start.year)
            for name in models:
                for t, scores in zip(TICKERS, model_scores(MODELS[name], data, splits, h)):
                    actual = data[t].turbulent[h].iloc[splits[t][1]].astype(bool).to_numpy()
                    rows.append(dict(model=name, h=h, ticker=t, year=start.year, n=len(scores),
                                     hits=int(((scores > 0.5) == actual).sum()), proba=list(scores), y=list(actual)))
    return pd.DataFrame(rows)


def summarize(results):
    """Accuracy, its standard error, AUC and per-ticker accuracy for each (horizon, model)."""
    out = []
    for (h, model), g in results.groupby(["h", "model"], sort=False):
        n = g.n.sum()
        accuracy = g.hits.sum() / n
        auc = roc_auc_score(sum(g.y, []), sum(g.proba, [])) if g.proba.notna().all() else np.nan
        per_ticker = (g.groupby("ticker").hits.sum() / g.groupby("ticker").n.sum()).round(3)
        out.append(dict(h=h, model=model, n=n, accuracy=accuracy, se=np.sqrt(accuracy * (1 - accuracy) / n),
                        auc=auc, **{f"acc_{t}": a for t, a in per_ticker.items()}))
    return pd.DataFrame(out).set_index(["h", "model"])


# ===========================================================================
# Live outlook for the dashboard
# ===========================================================================

# Measured accuracy of the live model ("Pooled logistic, no news") at h=5; see docs/VOLATILITY.md.
# Baseline = persistence ("same as the last 5 days") on the same test set.
LIVE_ACCURACY = {"validation_accuracy": 0.682, "test_accuracy": 0.764, "baseline_test_accuracy": 0.665}


def typical_moves(ticker, h):
    """Descriptive move sizes from the last 3 years of prices (not model outputs)."""
    returns = prices(ticker)["Close"].pct_change()
    week_vol = realised_vol(returns, h).iloc[-756:]
    turbulent = week_vol > realised_vol(returns, h).rolling(TRADING_DAYS_PER_YEAR).median().iloc[-756:]
    return {
        "normal_daily": float(realised_vol(returns, 20).rolling(TRADING_DAYS_PER_YEAR).median().iloc[-1]),
        "recent_daily": float(realised_vol(returns, 20).iloc[-1]),
        "turbulent_week_daily": float(week_vol[turbulent].mean()),
        "calm_week_daily": float(week_vol[~turbulent].mean()),
    }


def outlook(h=5):
    """Turbulence outlook for each ticker as of the latest close. Trains the pooled logistic model
    (no news features) on every day whose h-day outcome is already known."""
    data = {t: frame(t, news=False) for t in TICKERS}
    cols = feature_columns(data[TICKERS[0]].X, h, "no_news")
    known = [d.turbulent[h].notna() for d in data.values()]
    X_train = pd.concat([d.X[cols][k] for d, k in zip(data.values(), known)])
    y_train = pd.concat([d.turbulent[h][k].astype(bool) for d, k in zip(data.values(), known)])
    model = make_estimator("logistic").fit(X_train, y_train)

    out = {}
    for t, d in data.items():
        today = d.X[cols].iloc[[-1]]
        days_to = today.days_to_earn.iloc[0]
        out[t] = {
            "as_of": d.X.index[-1].strftime("%Y-%m-%d"),
            "horizon_days": h,
            "probability": float(model.predict_proba(today)[0, 1]),
            "moves": typical_moves(t, h),
            "drivers": {
                "vix_vs_normal": float(np.exp(today.vix.iloc[0]) - 1),  # features are logs of ratios
                "recent_vol_vs_normal": float(np.exp(today.rv20.iloc[0]) - 1),
                "earnings_in_window": bool(today[f"earn_in_{h}"].iloc[0]),
                "days_to_earnings": int(days_to) if days_to < 70 else None,  # 70 = capped / none known
            },
            **LIVE_ACCURACY,
        }
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", nargs=2, metavar=("MODEL", "HORIZON"), help="score one model once on 2022-2023")
    args = parser.parse_args()
    pd.set_option("display.width", 220)
    if args.test:
        results = run([args.test[0]], [int(args.test[1])], test=True)
    else:
        results = run(list(MODELS), HORIZONS)
        results.drop(columns=["proba", "y"]).to_csv(CACHE / "volatility_validation.csv", index=False)
    print(summarize(results).to_string(float_format="{:.3f}".format))
