"""direction experiments: can any model beat "always up"? (results: docs/EXPERIMENTS.md)

how it evaluates (walk-forward validation):
  for each year 2015–2021, train on everything before that year and forecast that year.
  the last `h` training rows are dropped ("purged") because their labels look into the
  scored year. forecasts are spaced `h` days apart so scored outcomes don't overlap.
  2022–2023 is the test set, used only with --test, once, for a model chosen in advance.

news features are point-in-time: the forecast for day t only uses headlines dated in the
7 days before t, and only matches past days whose outcome was already known on day t.

usage:
  uv run python scripts/experiments.py                                # validation, all models
  uv run --with sentence-transformers python scripts/experiments.py   # + embedding models
  uv run python scripts/experiments.py --target excess                # predict return minus SPY's
  uv run python scripts/experiments.py --test "HGB ext" 20            # one test run on 2022-2023
"""
import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNetCV, LogisticRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echomarket import CACHE, RIDGE_ALPHAS, TICKERS, dataset_news, prices, ridge_model, webhose_news

HORIZONS = (1, 5, 20)  # forecast 1, 5 and 20 trading days ahead
VAL_YEARS = range(2015, 2022)  # one validation fold per year
TEST = (pd.Timestamp("2022-01-01"), pd.Timestamp("2023-10-01"))  # news datasets end ~oct 2023
K = 15  # number of similar past news days to average, as on the dashboard

# feature sets
BASE = ["ret1", "mom5", "mom20", "vol20", "vol_ratio", "spy5"]  # the dashboard model's features
EXT = BASE + ["mom60", "vol60", "ma50_gap", "ma200_gap", "spy20"]  # extended price features
NEWS = ["news_count", "knn_tfidf", "knn_emb"]  # headline volume + news-similarity forecasts
LLM_NEWS = ["llm_sentiment", "llm_confidence"]  # optional, from scripts/llm_news.py


# ===========================================================================
# data
# ===========================================================================

def market(ticker, target="return"):
    """price features x (one row per trading day) and targets y (one column per horizon).
    target="return": the stock's forward return. target="excess": stock return minus spy's."""
    # the stock's prices and SPY (the whole market) on the same days.
    stock, spy = prices(ticker), prices("SPY")
    close, volume = stock["Close"], stock["Volume"]
    spy_close = spy["Close"].reindex(stock.index).ffill()
    daily_return = close.pct_change()
    # price features: recent returns, momentum, volatility, volume, trend and the market's moves.
    X = pd.DataFrame({
        "ret1": daily_return,
        "mom5": close.pct_change(5),
        "mom20": close.pct_change(20),
        "mom60": close.pct_change(60),
        "vol20": daily_return.rolling(20).std(),
        "vol60": daily_return.rolling(60).std(),
        "vol_ratio": volume / volume.rolling(20).mean(),
        "ma50_gap": close / close.rolling(50).mean() - 1,  # distance from the 50-day average
        "ma200_gap": close / close.rolling(200).mean() - 1,
        "spy5": spy_close.pct_change(5),
        "spy20": spy_close.pct_change(20),
    }).replace([np.inf, -np.inf], np.nan).dropna()
    X = X[X.index >= "2010-01-01"]  # the news datasets start in dec 2009

    # what we predict for each horizon h: the return over the next h days
    # (or, for "excess", how much the stock beat or lagged SPY).
    if target == "excess":
        Y = pd.DataFrame({h: close.shift(-h) / close - spy_close.shift(-h) / spy_close for h in HORIZONS})
    else:
        Y = pd.DataFrame({h: close.shift(-h) / close - 1 for h in HORIZONS})
    return X, Y.reindex(X.index)


def embeddings(titles):
    """sentence embedding (all-MiniLM-L6-v2) for each headline, cached on disk.
    returns none if the optional sentence-transformers package isn't installed."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    # only embed headlines we haven't embedded before; the rest come from the saved file.
    path = CACHE / "minilm_embeddings.parquet"
    cache = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    missing = sorted(set(titles) - set(cache.index))
    if missing:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        vectors = model.encode(missing, batch_size=256, normalize_embeddings=True, show_progress_bar=True)
        columns = [str(i) for i in range(vectors.shape[1])]
        cache = pd.concat([cache, pd.DataFrame(vectors.astype(np.float32), index=missing, columns=columns)])
        cache.to_parquet(path)
    return cache


def normalize_rows(matrix):
    """scale each row to length 1, so a dot product equals cosine similarity."""
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)


def load_news(ticker):
    """historical headlines for `ticker`, sorted by date."""
    news = pd.concat([dataset_news(ticker), webhose_news(ticker)], ignore_index=True).drop_duplicates("title")
    news["date"] = pd.to_datetime(news.date)
    news["title"] = news.title.astype(str)
    return news.sort_values("date", kind="stable").reset_index(drop=True)


def llm_features(ticker, news, queries, index):
    """average LLM sentiment/confidence of each day's query headlines (only if scores were generated)."""
    path = CACHE / "llm_news_scores.parquet"
    if not path.exists():
        return {}
    scores = pd.read_parquet(path)
    scores = scores[scores.ticker == ticker].drop_duplicates("title").set_index("title")
    out = {}
    for name in LLM_NEWS:
        per_headline = pd.Series(news.title.map(scores[name]).to_numpy(), index=news.index)
        daily = [per_headline.iloc[q].dropna().mean() for q in queries]  # NaN when nothing is scored
        out[name] = pd.Series(daily, index=index, dtype=float)
    return out


def knn_forecast(similarity, outcomes, n_eligible, queries):
    """for each day t: similarity-weighted average outcome of the k most similar eligible past days.
    `n_eligible[t]` = how many of the (date-sorted) memory days were usable on day t."""
    # for every day: look only at past days we were allowed to know about, pick the closest 15,
    # and average what happened after them (closer days count more).
    forecast = np.full(len(similarity), np.nan)
    for t in range(len(similarity)):
        m = n_eligible[t]
        if m == 0 or len(queries[t]) == 0:
            continue  # no usable history or no recent headlines
        sims = similarity[t, :m]
        top = np.argpartition(-sims, min(K, m) - 1)[:K]  # indices of the k largest similarities
        forecast[t] = np.average(outcomes[top], weights=np.maximum(sims[top], 1e-3))
    return forecast


def news_features(ticker, X, Y):
    """news features for every trading day in x:
      news_count               log(1 + headlines in the 7 days before t)
      (knn_tfidf, h)           kNN news forecast of the h-day outcome, TF-IDF similarity
      (knn_emb, h)             same with sentence embeddings (if installed)
      llm_sentiment/confidence optional LLM scores"""
    # all historical headlines for this stock, oldest first.
    news = load_news(ticker)
    news_dates = news.date.to_numpy()
    week_before = (X.index - pd.Timedelta(days=7)).to_numpy()

    # the "query" for day t = row numbers of the 10 newest headlines dated in [t - 7 days, t).
    first = np.searchsorted(news_dates, week_before)
    last = np.searchsorted(news_dates, X.index.to_numpy())
    queries = [np.arange(max(a, b - 10), b) for a, b in zip(first, last)]
    out = {"news_count": pd.Series(np.log1p(last - first), index=X.index)}
    out.update(llm_features(ticker, news, queries, X.index))

    # the "memory" = each trading day's own headlines, whose later outcome we can look up.
    rows_by_day = news[news.date.isin(X.index)].groupby("date").indices  # day -> headline row numbers
    memory_days = pd.DatetimeIndex(sorted(rows_by_day))
    memory_pos = X.index.get_indexer(memory_days)  # each memory day's row number in x

    # similarity between every query and every memory day.
    similarity = {}
    tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
    tfidf.fit(news.title[news.date < "2015-01-01"])  # vocabulary from pre-validation headlines only
    memory_docs = tfidf.transform([" ".join(news.title.iloc[rows_by_day[d]]) for d in memory_days])
    query_docs = tfidf.transform([" ".join(news.title.iloc[q]) for q in queries])
    similarity["knn_tfidf"] = (query_docs @ memory_docs.T).toarray().astype(np.float32)
    emb = embeddings(news.title)
    if emb is not None:
        E = emb.loc[news.title].to_numpy()
        memory_vecs = normalize_rows(np.stack([E[rows_by_day[d]].mean(0) for d in memory_days]))
        query_vecs = normalize_rows(np.stack([E[q].mean(0) if len(q) else np.zeros(E.shape[1]) for q in queries]))
        similarity["knn_emb"] = (query_vecs @ memory_vecs.T).astype(np.float32)

    for h in HORIZONS:
        outcomes = Y[h].to_numpy()[memory_pos]
        # memory days usable on day t: outcome finished by t (row <= t - h) AND older than the query week.
        # both conditions hold for a prefix of the date-sorted memory, so we only need its length.
        n_eligible = np.minimum(
            np.searchsorted(memory_pos, np.arange(len(X)) - h, side="right"),
            np.searchsorted(memory_days.to_numpy(), week_before),
        )
        for name, sims in similarity.items():
            out[(name, h)] = pd.Series(knn_forecast(sims, outcomes, n_eligible, queries), index=X.index)
    return out


# ===========================================================================
# models
# every fit_* function trains on (X_train, y_train) and returns one score per row of X_eval.
# the sign of the score is the predicted direction (for regressors it is the predicted return).
# `cols` picks the feature columns; columns that don't exist (e.g. knn_emb without
# sentence-transformers) are skipped.
# ===========================================================================

def available(cols, X):
    """keep only the columns that exist (knn_emb is missing without sentence-transformers)."""
    return [c for c in cols if c in X]


def fit_always_up(X_train, y_train, X_eval):
    """baseline: predict the training-period average return (almost always positive = 'up')."""
    return np.full(len(X_eval), y_train.mean())


def fit_dashboard_ridge(X_train, y_train, X_eval):
    """the dashboard's model: ridge(alpha=1) on the 6 BASE features."""
    return ridge_model(1.0).fit(X_train[BASE], y_train).predict(X_eval[BASE])


def fit_ridge(X_train, y_train, X_eval, cols):
    """ridge regression; alpha chosen by cross-validation. missing news values are filled with 0."""
    c = available(cols, X_train)
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    return model.fit(X_train[c].fillna(0), y_train).predict(X_eval[c].fillna(0))


def fit_elastic_net(X_train, y_train, X_eval, cols):
    """ElasticNet (mix of lasso and ridge); strength chosen by time-ordered cross-validation.
    the target is rescaled to unit variance so the alpha grid fits any horizon."""
    c = available(cols, X_train)
    scale = max(float(y_train.std()), 1e-8)
    model = make_pipeline(
        StandardScaler(),
        ElasticNetCV(l1_ratio=(0.1, 0.5, 0.9, 1.0), alphas=np.logspace(-3, 0, 13),
                     cv=TimeSeriesSplit(n_splits=5, gap=max(HORIZONS)), max_iter=20000),
    )
    return model.fit(X_train[c].fillna(0), y_train / scale).predict(X_eval[c].fillna(0)) * scale


def fit_boosted_trees(X_train, y_train, X_eval, cols):
    """gradient-boosted trees (HGB), kept shallow and regularised. handles missing values natively."""
    c = available(cols, X_train)
    model = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                          min_samples_leaf=50, l2_regularization=1.0, random_state=0)
    return model.fit(X_train[c], y_train).predict(X_eval[c])


def fit_forest(X_train, y_train, X_eval, cols, extra_trees=False):
    """random forest, or extra trees (more randomised splits)."""
    c = available(cols, X_train)
    Forest = ExtraTreesRegressor if extra_trees else RandomForestRegressor
    model = Forest(n_estimators=200, min_samples_leaf=20, max_features=0.7, n_jobs=-1, random_state=0)
    return model.fit(X_train[c].fillna(0), y_train).predict(X_eval[c].fillna(0))


def fit_logistic(X_train, y_train, X_eval):
    """logistic regression on direction (up/down). score = p(up) - 0.5."""
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
    return model.fit(X_train[EXT], y_train > 0).predict_proba(X_eval[EXT])[:, 1] - 0.5


def fit_boosted_classifier(X_train, y_train, X_eval):
    """gradient-boosted trees classifying direction. score = p(up) - 0.5."""
    model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                           min_samples_leaf=50, l2_regularization=1.0, random_state=0)
    return model.fit(X_train[EXT], y_train > 0).predict_proba(X_eval[EXT])[:, 1] - 0.5


def fit_news_knn(X_train, y_train, X_eval, col):
    """use a precomputed kNN news forecast directly; without recent news, predict the training mean."""
    return X_eval[col].fillna(y_train.mean()).to_numpy()


def fit_gated(X_train, y_train, X_eval, model, quantile=0.2):
    """wraps another model: says 'down' only when its score is in the most bearish 20% of the
    scores it gives on the training period, and 'up' otherwise."""
    scores = np.asarray(model(X_train, y_train, pd.concat([X_train, X_eval])))
    train_scores, eval_scores = scores[: len(X_train)], scores[len(X_train):]
    return eval_scores - np.quantile(train_scores, quantile)


# name -> function(X_train, y_train, X_eval). partial() fills in each model's settings.
MODELS = {
    "Always up (train mean)": fit_always_up,
    "Ridge base (current)": fit_dashboard_ridge,
    "Ridge ext": partial(fit_ridge, cols=EXT),
    "ElasticNet ext": partial(fit_elastic_net, cols=EXT),
    "Random forest ext": partial(fit_forest, cols=EXT),
    "Extra trees ext": partial(fit_forest, cols=EXT, extra_trees=True),
    "Logistic ext": fit_logistic,
    "HGB ext": partial(fit_boosted_trees, cols=EXT),
    "HGB classifier ext": fit_boosted_classifier,
    "News kNN TF-IDF": partial(fit_news_knn, col="knn_tfidf"),
    "News kNN embeddings": partial(fit_news_knn, col="knn_emb"),
    "Ridge ext + news": partial(fit_ridge, cols=EXT + NEWS),
    "HGB ext + news": partial(fit_boosted_trees, cols=EXT + NEWS),
    "Ridge ext + LLM news": partial(fit_ridge, cols=EXT + LLM_NEWS),
    "ElasticNet ext + LLM news": partial(fit_elastic_net, cols=EXT + LLM_NEWS),
    "HGB ext + LLM news": partial(fit_boosted_trees, cols=EXT + LLM_NEWS),
    "Ridge ext + news, gated": partial(fit_gated, model=partial(fit_ridge, cols=EXT + NEWS)),
    "Logistic ext, gated": partial(fit_gated, model=fit_logistic),
    "News kNN embeddings, gated": partial(fit_gated, model=partial(fit_news_knn, col="knn_emb")),
}


# ===========================================================================
# evaluation
# ===========================================================================

def periods(test):
    """(start, end) of each scored period: one per validation year, or the single test period."""
    if test:
        return [TEST]
    return [(pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")) for y in VAL_YEARS]


def can_run(name, features):
    """skip models whose optional inputs weren't generated."""
    if "embeddings" in name and "knn_emb" not in features:
        return False  # sentence-transformers not installed
    if "+ LLM news" in name and not set(LLM_NEWS).issubset(features.columns):
        return False  # scripts/llm_news.py hasn't been run
    return True


def run(models, horizons, test=False, target="return"):
    """train and score every model on every ticker, horizon and period. one result row per combination."""
    rows = []
    tickers = [t for t in TICKERS if not (target == "excess" and t == "SPY")]  # SPY minus SPY is 0
    for ticker in tickers:
        # prices, targets and news features for this stock (built once, reused for every model).
        X, Y = market(ticker, target)
        nf = news_features(ticker, X, Y)
        for h in horizons:
            # feature table for this horizon (the kNN forecasts depend on h).
            features = X.assign(news_count=nf["news_count"], knn_tfidf=nf[("knn_tfidf", h)])
            for name in LLM_NEWS:
                if name in nf:
                    features[name] = nf[name]
            if ("knn_emb", h) in nf:
                features["knn_emb"] = nf[("knn_emb", h)]
            y = Y[h]
            has_label = y.notna().to_numpy()

            for start, end in periods(test):
                train = np.flatnonzero((X.index < start) & has_label)[:-h]  # purge the last h rows
                scored = np.flatnonzero((X.index >= start) & (X.index < end) & has_label)[::h]  # every h-th day
                actual = y.iloc[scored].to_numpy()
                # train each model on the training rows and score it on the scored rows.
                for name in models:
                    if not can_run(name, features):
                        continue
                    fit = MODELS[name]
                    pred = np.asarray(fit(features.iloc[train], y.iloc[train], features.iloc[scored]), dtype=float)
                    constant = np.ptp(pred) == 0  # a constant forecast has no rank correlation
                    rows.append({
                        "ticker": ticker, "h": h, "year": start.year, "model": name, "n": len(scored),
                        "hits": int(((pred > 0) == (actual > 0)).sum()),  # correct direction calls
                        "ups": int((actual > 0).sum()),  # what "always up" would get right
                        "ic": np.nan if constant else pd.Series(pred).corr(pd.Series(actual), method="spearman"),
                    })
    return pd.DataFrame(rows)


def summarize(results):
    """one line per (horizon, model): hit rate vs always-up, and rank correlation (IC)."""
    # a constant forecast (e.g. ElasticNet shrunk to the mean) has no rank information, so count its
    # IC as 0. averaging only the non-constant periods would inflate mean_IC.
    constant = results.ic.isna() & (results.model != "Always up (train mean)")
    results = results.assign(constant=constant, ic=results.ic.mask(constant, 0.0),
                             beat_up=results.hits > results.ups)
    g = results.groupby(["h", "model"], sort=False)
    s = pd.DataFrame({
        "n": g.n.sum(),
        "direction": g.hits.sum() / g.n.sum(),
        "always_up": g.ups.sum() / g.n.sum(),
        "mean_IC": g.ic.mean(),
        "constant_folds": g.constant.mean(),
        "IC_t": g.ic.mean() / (g.ic.std() / np.sqrt(g.ic.count())),  # mean IC / its standard error
        "years_beating_up": g.beat_up.mean(),
    })
    s["edge_pts"] = (s.direction - s.always_up) * 100
    s["se_pts"] = np.sqrt(s.always_up * (1 - s.always_up) / s.n) * 100  # ~1 standard error of a hit rate
    return s[["n", "direction", "always_up", "edge_pts", "se_pts", "mean_IC", "IC_t", "constant_folds",
              "years_beating_up"]]


FORMATS = {"direction": "{:.3f}", "always_up": "{:.3f}", "edge_pts": "{:+.1f}", "se_pts": "{:.1f}",
           "mean_IC": "{:+.3f}", "IC_t": "{:+.1f}", "constant_folds": "{:.0%}", "years_beating_up": "{:.0%}"}

# command line: validation for every model, or one --test run on 2022-2023.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("return", "excess"), default="return",
                        help="excess = beat SPY (NVDA/TSLA/AAPL only)")
    parser.add_argument("--test", nargs=2, metavar=("MODEL", "HORIZON"), help="score one model once on 2022-2023")
    args = parser.parse_args()
    pd.set_option("display.width", 200)
    if args.test:
        model, h = args.test[0], int(args.test[1])
        results = run(["Always up (train mean)", "Ridge base (current)", model], [h], test=True, target=args.target)
        print(results.to_string())
    else:
        results = run(list(MODELS), HORIZONS, target=args.target)
        results.to_csv(CACHE / f"experiments_validation_{args.target}.csv", index=False)
    print(summarize(results).to_string(formatters={k: v.format for k, v in FORMATS.items()}))
