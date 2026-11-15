"""Model experiments: walk-forward validation over several years and horizons.

Validation: for each year 2015-2021, train on everything before it (last `h` rows purged so
no label crosses the boundary) and score that year with non-overlapping forecasts
(one every `h` trading days). 2022+ is the test set and is only scored with --test.

News features are point-in-time: the query for day t uses headlines dated in the 7 days
strictly before t, and matched past days must have a finished h-day outcome by t.

Usage:
  uv run python scripts/experiments.py                      # validation, all models
  uv run --with sentence-transformers python scripts/experiments.py   # + embedding models
  uv run python scripts/experiments.py --target excess      # predict return minus SPY's
  uv run python scripts/experiments.py --test "HGB ext" 20  # score one model once on 2022-2023
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LogisticRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echomarket import CACHE, RIDGE_ALPHAS, TICKERS, dataset_news, prices, ridge_model, webhose_news

HORIZONS = (1, 5, 20)
VAL_YEARS = range(2015, 2022)
TEST = (pd.Timestamp("2022-01-01"), pd.Timestamp("2023-10-01"))  # news datasets end ~Oct 2023
K = 15  # matched days, as on the dashboard
BASE = ["ret1", "mom5", "mom20", "vol20", "vol_ratio", "spy5"]  # the current model's features
EXT = BASE + ["mom60", "vol60", "ma50_gap", "ma200_gap", "spy20"]
NEWS = ["news_count", "knn_tfidf", "knn_emb"]
LLM_NEWS = ["llm_sentiment", "llm_confidence"]


# ---------- data ----------

def market(ticker, target="return"):
    d, spy = prices(ticker), prices("SPY")
    c, v, s = d["Close"], d["Volume"], spy["Close"].reindex(d.index).ffill()
    r = c.pct_change()
    X = pd.DataFrame(
        {
            "ret1": r, "mom5": c.pct_change(5), "mom20": c.pct_change(20), "mom60": c.pct_change(60),
            "vol20": r.rolling(20).std(), "vol60": r.rolling(60).std(), "vol_ratio": v / v.rolling(20).mean(),
            "ma50_gap": c / c.rolling(50).mean() - 1, "ma200_gap": c / c.rolling(200).mean() - 1,
            "spy5": s.pct_change(5), "spy20": s.pct_change(20),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    X = X[X.index >= "2010-01-01"]  # news starts Dec 2009
    if target == "excess":  # stock return minus SPY return: removes the market's upward drift
        Y = pd.DataFrame({h: c.shift(-h) / c - s.shift(-h) / s for h in HORIZONS})
    else:
        Y = pd.DataFrame({h: c.shift(-h) / c - 1 for h in HORIZONS})
    Y = Y.reindex(X.index)
    return X, Y


def embeddings(titles):
    """title -> unit vector (all-MiniLM-L6-v2), cached; None if sentence-transformers isn't installed."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    path = CACHE / "minilm_embeddings.parquet"
    cache = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    missing = sorted(set(titles) - set(cache.index))
    if missing:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        e = model.encode(missing, batch_size=256, normalize_embeddings=True, show_progress_bar=True)
        new = pd.DataFrame(e.astype(np.float32), index=missing, columns=[str(i) for i in range(e.shape[1])])
        cache = pd.concat([cache, new])
        cache.to_parquet(path)
    return cache


def unit(m):
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


def news_features(ticker, X, Y):
    """Per trading day: headline count in the prior 7 days, and kNN news forecasts per horizon."""
    news = pd.concat([dataset_news(ticker), webhose_news(ticker)], ignore_index=True).drop_duplicates("title")
    news["date"] = pd.to_datetime(news.date)
    news = news.sort_values("date", kind="stable").reset_index(drop=True)
    news["title"] = news.title.astype(str)
    nd = news.date.to_numpy()

    # Query = the 10 newest headlines in [t-7d, t).
    lo = np.searchsorted(nd, (X.index - pd.Timedelta(days=7)).to_numpy())
    hi = np.searchsorted(nd, X.index.to_numpy())
    queries = [np.arange(max(a, b - 10), b) for a, b in zip(lo, hi)]
    out = {"news_count": pd.Series(np.log1p(hi - lo), index=X.index)}

    # Optional headline scores produced by scripts/llm_news.py. Scores are joined
    # by ticker and title, then aggregated only over headlines available before t.
    llm_path = CACHE / "llm_news_scores.parquet"
    if llm_path.exists():
        llm = pd.read_parquet(llm_path)
        llm = llm[llm.ticker == ticker].drop_duplicates("title").set_index("title")
        scored = news.title.map(llm.llm_sentiment), news.title.map(llm.llm_confidence)
        for name, values in zip(LLM_NEWS, scored):
            values = pd.Series(values.to_numpy(), index=news.index)
            daily = []
            for q in queries:
                valid = values.iloc[q].dropna()
                daily.append(float(valid.mean()) if len(valid) else np.nan)
            out[name] = pd.Series(daily, index=X.index)

    # Memory = each trading day's own headlines, with its realised outcome.
    on_trading_day = news[news.date.isin(X.index)]
    groups = on_trading_day.groupby("date").indices  # day -> row numbers in news
    days = pd.DatetimeIndex(sorted(groups))
    day_pos = X.index.get_indexer(days)

    sims = {}
    vec = TfidfVectorizer(max_features=3000, stop_words="english").fit(news.title[news.date < "2015-01-01"])
    T = vec.transform(news.title)
    D = vec.transform([" ".join(news.title.iloc[groups[d]]) for d in days])
    Q = vec.transform([" ".join(news.title.iloc[q]) for q in queries])
    sims["knn_tfidf"] = (Q @ D.T).toarray().astype(np.float32)
    emb = embeddings(news.title)
    if emb is not None:
        E = emb.loc[news.title].to_numpy()
        De = unit(np.stack([E[groups[d]].mean(0) for d in days]))
        Qe = unit(np.stack([E[q].mean(0) if len(q) else np.zeros(E.shape[1]) for q in queries]))
        sims["knn_emb"] = (Qe @ De.T).astype(np.float32)

    for h in HORIZONS:
        outcomes = Y[h].to_numpy()[day_pos]
        # Eligible memory is a prefix of `days`: outcome finished by t and older than the query window.
        m = np.minimum(
            np.searchsorted(day_pos, np.arange(len(X)) - h, side="right"),
            np.searchsorted(days.to_numpy(), (X.index - pd.Timedelta(days=7)).to_numpy()),
        )
        for name, S in sims.items():
            pred = np.full(len(X), np.nan)
            for t in range(len(X)):
                if m[t] == 0 or len(queries[t]) == 0:
                    continue
                s = S[t, : m[t]]
                top = np.argpartition(-s, min(K, m[t]) - 1)[:K]
                pred[t] = np.average(outcomes[top], weights=np.maximum(s[top], 1e-3))
            out[(name, h)] = pd.Series(pred, index=X.index)
    return out


# ---------- models ----------
# Each returns a score whose sign is the predicted direction (regressors: predicted return).

def linear(cols):
    def fit(Xtr, ytr, Xev):
        c = [x for x in cols if x in Xtr]  # knn_emb only exists with sentence-transformers
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
        return m.fit(Xtr[c].fillna(0), ytr).predict(Xev[c].fillna(0))
    return fit


def elastic_net(cols):
    def fit(Xtr, ytr, Xev):
        c = [x for x in cols if x in Xtr]
        y_scale = max(float(ytr.std()), 1e-8)
        m = make_pipeline(
            StandardScaler(),
            ElasticNetCV(
                l1_ratio=(0.1, 0.5, 0.9, 1.0),
                alphas=np.logspace(-3, 0, 13),
                cv=TimeSeriesSplit(n_splits=5, gap=max(HORIZONS)),
                max_iter=20000,
            ),
        )
        return m.fit(Xtr[c].fillna(0), ytr / y_scale).predict(Xev[c].fillna(0)) * y_scale
    return fit


def hgb(cols):
    def fit(Xtr, ytr, Xev):
        c = [x for x in cols if x in Xtr]
        m = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50, l2_regularization=1.0, random_state=0
        )
        return m.fit(Xtr[c], ytr).predict(Xev[c])
    return fit


def forest(kind, cols):
    def fit(Xtr, ytr, Xev):
        c = [x for x in cols if x in Xtr]
        estimator = (RandomForestRegressor if kind == "random_forest" else ExtraTreesRegressor)(
            n_estimators=200, min_samples_leaf=20, max_features=0.7, n_jobs=-1, random_state=0
        )
        return estimator.fit(Xtr[c].fillna(0), ytr).predict(Xev[c].fillna(0))
    return fit


def logit(Xtr, ytr, Xev):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)).fit(Xtr[EXT], ytr > 0)
    return m.predict_proba(Xev[EXT])[:, 1] - 0.5


def hgb_clf(Xtr, ytr, Xev):
    m = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50, l2_regularization=1.0, random_state=0
    ).fit(Xtr[EXT], ytr > 0)
    return m.predict_proba(Xev[EXT])[:, 1] - 0.5


def knn(col):
    # Standalone news forecast; fall back to the training mean when there is no recent news.
    return lambda Xtr, ytr, Xev: Xev[col].fillna(ytr.mean()).to_numpy()


def gated(fn, q=0.2):
    """Say 'down' only when the model is in its most bearish q of training-period predictions."""
    def fit(Xtr, ytr, Xev):
        p = np.asarray(fn(Xtr, ytr, pd.concat([Xtr, Xev])))
        return p[len(Xtr):] - np.quantile(p[: len(Xtr)], q)
    return fit


MODELS = {
    "Always up (train mean)": lambda Xtr, ytr, Xev: np.full(len(Xev), ytr.mean()),
    "Ridge base (current)": lambda Xtr, ytr, Xev: ridge_model(1.0).fit(Xtr[BASE], ytr).predict(Xev[BASE]),
    "Ridge ext": linear(EXT),
    "ElasticNet ext": elastic_net(EXT),
    "Random forest ext": forest("random_forest", EXT),
    "Extra trees ext": forest("extra_trees", EXT),
    "Logistic ext": logit,
    "HGB ext": hgb(EXT),
    "HGB classifier ext": hgb_clf,
    "News kNN TF-IDF": knn("knn_tfidf"),
    "News kNN embeddings": knn("knn_emb"),
    "Ridge ext + news": linear(EXT + NEWS),
    "HGB ext + news": hgb(EXT + NEWS),
    "Ridge ext + LLM news": linear(EXT + LLM_NEWS),
    "ElasticNet ext + LLM news": elastic_net(EXT + LLM_NEWS),
    "HGB ext + LLM news": hgb(EXT + LLM_NEWS),
    "Ridge ext + news, gated": gated(linear(EXT + NEWS)),
    "Logistic ext, gated": gated(logit),
    "News kNN embeddings, gated": gated(knn("knn_emb")),
}


# ---------- evaluation ----------

def folds(index, h, test):
    if test:
        yield TEST
    else:
        for y in VAL_YEARS:
            yield pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")


def run(models, horizons, test=False, target="return"):
    rows = []
    for ticker in [t for t in TICKERS if not (target == "excess" and t == "SPY")]:
        X, Y = market(ticker, target)
        nf = news_features(ticker, X, Y)
        for h in horizons:
            F = X.assign(news_count=nf["news_count"], knn_tfidf=nf[("knn_tfidf", h)])
            for name in LLM_NEWS:
                if name in nf:
                    F[name] = nf[name]
            if ("knn_emb", h) in nf:
                F["knn_emb"] = nf[("knn_emb", h)]
            y = Y[h]
            for start, end in folds(X.index, h, test):
                tr = np.flatnonzero((X.index < start) & y.notna().to_numpy())[:-h]  # purge h rows
                ev = np.flatnonzero((X.index >= start) & (X.index < end) & y.notna().to_numpy())[::h]
                actual = y.iloc[ev].to_numpy()
                for name in models:
                    if "embeddings" in name and "knn_emb" not in F:
                        continue  # sentence-transformers not installed
                    if "+ LLM news" in name and not set(LLM_NEWS).issubset(F.columns):
                        continue  # LLM scores have not been generated
                    pred = np.asarray(MODELS[name](F.iloc[tr], y.iloc[tr], F.iloc[ev]), dtype=float)
                    rows.append({
                        "ticker": ticker, "h": h, "year": start.year, "model": name, "n": len(ev),
                        "hits": int(((pred > 0) == (actual > 0)).sum()), "ups": int((actual > 0).sum()),
                        "ic": pd.Series(pred).corr(pd.Series(actual), method="spearman") if np.ptp(pred) > 0 else np.nan,
                    })
    return pd.DataFrame(rows)


def summarize(r):
    g = r.groupby(["h", "model"], sort=False)
    s = pd.DataFrame({
        "n": g.n.sum(),
        "direction": g.hits.sum() / g.n.sum(),
        "always_up": g.ups.sum() / g.n.sum(),
        "mean_IC": g.ic.mean(),
        "IC_t": g.ic.mean() / (g.ic.std() / np.sqrt(g.ic.count())),
        "years_beating_up": r.assign(win=r.hits > r.ups).groupby(["h", "model"], sort=False).win.mean(),
    })
    s["edge_pts"] = (s.direction - s.always_up) * 100
    s["se_pts"] = np.sqrt(s.always_up * (1 - s.always_up) / s.n) * 100  # ~1 std error of a hit rate
    return s[["n", "direction", "always_up", "edge_pts", "se_pts", "mean_IC", "IC_t", "years_beating_up"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=("return", "excess"), default="return",
                    help="excess = beat SPY (NVDA/TSLA/AAPL only)")
    ap.add_argument("--test", nargs=2, metavar=("MODEL", "HORIZON"), help="score one model once on 2022-2023")
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    fmt = {"direction": "{:.3f}", "always_up": "{:.3f}", "edge_pts": "{:+.1f}", "se_pts": "{:.1f}",
           "mean_IC": "{:+.3f}", "IC_t": "{:+.1f}", "years_beating_up": "{:.0%}"}
    if a.test:
        model, h = a.test[0], int(a.test[1])
        r = run(["Always up (train mean)", "Ridge base (current)", model], [h], test=True, target=a.target)
        print(r.to_string())
    else:
        r = run(list(MODELS), HORIZONS, target=a.target)
        r.to_csv(CACHE / f"experiments_validation_{a.target}.csv", index=False)
    s = summarize(r)
    print(s.to_string(formatters={k: v.format for k, v in fmt.items()}))
