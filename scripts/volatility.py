"""Volatility experiments: will the next h trading days be more turbulent than usual?

Target: realised volatility over the next h days, sqrt(mean(r^2)), above the stock's
trailing 1-year median of h-day volatility, which is known at t. Up or down doesn't matter.
Baselines: majority class, and persistence ("same as the last h days").

Validation (2015-2021) and test (2022-01 to 2023-09) work as in experiments.py. Features use
only information available at the close of day t: earnings dates are scheduled in advance,
and news is from the 7 days strictly before t.

Usage:
  uv run python scripts/volatility.py                     # walk-forward validation, all models
  uv run python scripts/volatility.py --test "Pooled HGB" 5  # score one model once on the test set
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echomarket import CACHE, TICKERS, prices
from experiments import TEST, VAL_YEARS, news_features

HORIZONS = (1, 5, 20)


def rv(r, w):
    """Root-mean-square daily return over a trailing window of w days."""
    return np.sqrt((r**2).rolling(w).mean())


def earnings_reaction_days(ticker, index):
    """Trading days on which the market first reacts to each earnings report (none for SPY)."""
    if ticker == "SPY":
        return pd.DatetimeIndex([])
    p = CACHE / f"earnings_{ticker}.parquet"
    if p.exists():
        stamps = pd.read_parquet(p).date
    else:
        e = yf.Ticker(ticker).get_earnings_dates(limit=100)
        stamps = pd.Series(e.index.tz_convert("America/New_York").tz_localize(None), name="date")
        stamps.to_frame().to_parquet(p)
    stamps = pd.to_datetime(stamps)
    # After-close reports (16:00) move the stock the next session.
    day = stamps.dt.normalize() + pd.to_timedelta((stamps.dt.hour >= 16).astype(int), unit="D")
    pos = index.searchsorted(pd.DatetimeIndex(day).as_unit(index.unit))  # next trading day on/after `day`
    return pd.DatetimeIndex(index[pos[pos < len(index)]])


def frame(ticker):
    d, spy, vix = prices(ticker), prices("SPY"), prices("^VIX")
    c = d["Close"]
    r = c.pct_change()
    rs = spy["Close"].reindex(d.index).ffill().pct_change()
    v = vix["Close"].reindex(d.index).ffill()
    pos = np.arange(len(d))
    earn = np.unique(d.index.searchsorted(earnings_reaction_days(ticker, d.index)))
    earn = np.append(earn, 10**9)  # sentinel: no further earnings (SPY has none at all)
    nxt = np.searchsorted(earn, pos, side="right")  # first reaction day after t
    days_to = earn[nxt] - pos
    prev = earn[np.maximum(nxt - 1, 0)]
    days_since = np.where((nxt > 0) & (prev <= pos), pos - prev, 999)

    X = pd.DataFrame(index=d.index)
    targets, cont, persist = {}, {}, {}
    for h in HORIZONS:
        past = rv(r, h)
        thr = past.rolling(252).median()
        fwd = past.shift(-h)  # rv over t+1..t+h
        targets[h] = (fwd > thr).where(fwd.notna())
        cont[h] = np.log((fwd + 1e-6) / (thr + 1e-6))  # a flat day has rv = 0
        persist[h] = past > thr
        X[f"earn_in_{h}"] = ((days_to >= 1) & (days_to <= h)).astype(float)
    thr20 = rv(r, 20).rolling(252).median()
    for w in (1, 5, 20, 60):
        X[f"rv{w}"] = np.log(rv(r, w) / thr20 + 1e-6)
    X["spy_rv20"] = np.log(rv(rs, 20) / rv(rs, 20).rolling(252).median())
    X["vix"] = np.log(v / v.rolling(252).median())
    X["vix_chg5"] = np.log(v / v.shift(5))
    X["days_to_earn"] = np.minimum(days_to, 70)
    X["days_since_earn"] = np.minimum(days_since, 70)
    X["ret1"], X["mom20"] = r, c.pct_change(20)
    X["ma200_gap"] = c / c.rolling(200).mean() - 1
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    X = X[X.index >= "2011-01-01"]
    Y = pd.DataFrame(targets).reindex(X.index)
    C = pd.DataFrame(cont).reindex(X.index)
    P = pd.DataFrame(persist).reindex(X.index)

    # News: headline volume, and the kNN-average volatility that followed similar headline days.
    nf = news_features(ticker, X, C)
    X["news_count"] = nf["news_count"]
    for h in HORIZONS:
        X[f"knn_vol_{h}"] = nf[("knn_tfidf", h)]
    return X.replace([np.inf, -np.inf], np.nan), Y, P


def cols(X, h, news=True):
    c = [x for x in X if not x.startswith(("earn_in_", "knn_vol_"))] + [f"earn_in_{h}"]
    if news:
        c += [f"knn_vol_{h}"]
    else:
        c.remove("news_count")
    return c


def clf(kind):
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                              min_samples_leaf=50, l2_regularization=1.0, random_state=0)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=30, max_features=0.5, n_jobs=-1, random_state=0)
    if kind == "et":
        return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=30, max_features=0.5, n_jobs=-1, random_state=0)


def fit_proba(kind, Xtr, ytr, Xev):
    if kind == "ensemble":
        return np.mean([fit_proba(k, Xtr, ytr, Xev) for k in ("logit", "hgb", "rf")], axis=0)
    fill = Xtr.median()  # knn_vol is NaN on days with no recent news
    return clf(kind).fit(Xtr.fillna(fill), ytr).predict_proba(Xev.fillna(fill))[:, 1]


# name -> (classifier kind, use news, pooled across tickers)
MODELS = {
    "VIX-only logistic": ("vix", False, False),
    "Logistic": ("logit", True, False),
    "HGB": ("hgb", True, False),
    "HGB, no news": ("hgb", False, False),
    "Random forest": ("rf", True, False),
    "Extra trees": ("et", True, False),
    "Ensemble (logit+HGB+RF)": ("ensemble", True, False),
    "Pooled logistic": ("logit", True, True),
    "Pooled HGB": ("hgb", True, True),
    "Pooled random forest": ("rf", True, True),
    "Pooled ensemble": ("ensemble", True, True),
}


def run(models, horizons, test=False):
    data = {t: frame(t) for t in TICKERS}
    rows = []
    for h in horizons:
        for start, end in [TEST] if test else [(pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")) for y in VAL_YEARS]:
            split = {}
            for t, (X, Y, P) in data.items():
                ok = Y[h].notna().to_numpy()
                tr = np.flatnonzero((X.index < start) & ok)[:-h]  # purge h rows
                ev = np.flatnonzero((X.index >= start) & (X.index < end) & ok)[::h]  # non-overlapping
                split[t] = (tr, ev)
            for t, (X, Y, P) in data.items():  # baselines, per ticker
                tr, ev = split[t]
                y_ev = Y[h].iloc[ev].astype(bool).to_numpy()
                maj = Y[h].iloc[tr].mean() > 0.5
                for name, guess in [("Majority class", np.full(len(ev), maj)),
                                    ("Persistence (last h days)", P[h].iloc[ev].to_numpy())]:
                    rows.append(dict(model=name, h=h, ticker=t, year=start.year, n=len(ev),
                                     hits=int((guess == y_ev).sum()), proba=None, y=None))
            for name in models:
                kind, news, pooled = MODELS[name]
                c = ["vix", "spy_rv20", "rv20"] if kind == "vix" else cols(data[TICKERS[0]][0], h, news)
                kind = "logit" if kind == "vix" else kind
                ev_frames = [data[t][0][c].iloc[split[t][1]] for t in TICKERS]
                if pooled:  # one model trained on every ticker's training rows, fit once
                    Xtr = pd.concat([data[t][0][c].iloc[split[t][0]] for t in TICKERS])
                    ytr = pd.concat([data[t][1][h].iloc[split[t][0]] for t in TICKERS]).astype(bool)
                    probs = np.split(fit_proba(kind, Xtr, ytr, pd.concat(ev_frames)),
                                     np.cumsum([len(e) for e in ev_frames])[:-1])
                else:
                    probs = [fit_proba(kind, data[t][0][c].iloc[split[t][0]],
                                       data[t][1][h].iloc[split[t][0]].astype(bool), e)
                             for t, e in zip(TICKERS, ev_frames)]
                for t, p in zip(TICKERS, probs):
                    y_ev = data[t][1][h].iloc[split[t][1]].astype(bool).to_numpy()
                    rows.append(dict(model=name, h=h, ticker=t, year=start.year, n=len(p),
                                     hits=int(((p > 0.5) == y_ev).sum()), proba=list(p), y=list(y_ev)))
    return pd.DataFrame(rows)


def summarize(r):
    out = []
    for (h, m), g in r.groupby(["h", "model"], sort=False):
        acc = g.hits.sum() / g.n.sum()
        auc = np.nan
        if g.proba.notna().all():
            auc = roc_auc_score(sum(g.y, []), sum(g.proba, []))
        per_ticker = (g.groupby("ticker").hits.sum() / g.groupby("ticker").n.sum()).round(3).to_dict()
        out.append(dict(h=h, model=m, n=g.n.sum(), accuracy=acc, se=np.sqrt(acc * (1 - acc) / g.n.sum()), auc=auc,
                        **{f"acc_{k}": v for k, v in per_ticker.items()}))
    return pd.DataFrame(out).set_index(["h", "model"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", nargs=2, metavar=("MODEL", "HORIZON"))
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    if a.test:
        r = run([a.test[0]], [int(a.test[1])], test=True)
    else:
        r = run(list(MODELS), HORIZONS)
        r.drop(columns=["proba", "y"]).to_csv(CACHE / "volatility_validation.csv", index=False)
    print(summarize(r).to_string(float_format=lambda x: f"{x:.3f}"))
