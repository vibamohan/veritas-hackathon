"""Does the news half of the model help? Point-in-time validation on 2020-2021.

Same split as /api/replay: train through 2019, validate 2020-2021 (every 5 trading
days), five-row purge at boundaries, 2022+ untouched. For each validation date t:
  - "today's story" = the 10 newest headlines from the 7 days before t (strictly before t,
    so nothing published after the forecast is made),
  - candidates = past days whose 5-day outcome had finished by t,
  - TF-IDF vocabulary is fit on pre-2020 headlines only.

Usage: uv run python scripts/evaluate.py [TICKER ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echomarket import FEATURES, TICKERS, dataset_news, market_frame, ridge_model, webhose_news

TRAIN_END, VAL_END = pd.Timestamp("2020-01-01"), pd.Timestamp("2022-01-01")
TOP_K, NEWS_WEIGHT, H = 15, 0.3, 5  # same constants as forecast_bundle


def score(pred, actual):
    pred, actual = np.asarray(pred), np.asarray(actual)
    hit = [np.sign(p) == np.sign(a) for p, a in zip(pred, actual) if p != 0]
    corr = np.corrcoef(pred, actual)[0, 1] if np.ptp(pred) > 0 else np.nan  # undefined for a constant
    return {"MAE": np.abs(pred - actual).mean(), "direction": np.mean(hit), "corr": corr}


def evaluate(ticker):
    f = market_frame(ticker).dropna()
    pos = {d: i for i, d in enumerate(f.index)}
    train = np.flatnonzero(f.index < TRAIN_END)[: -H]
    val = np.flatnonzero((f.index >= TRAIN_END) & (f.index < VAL_END))[H:-H:H]
    model = ridge_model(1.0).fit(f[FEATURES].iloc[train], f.target5.iloc[train])
    price = model.predict(f[FEATURES].iloc[val])

    # Historical news only (GDELT is live-only, so it can't be replayed).
    news = pd.concat([dataset_news(ticker), webhose_news(ticker)], ignore_index=True).drop_duplicates("title")
    news["date"] = pd.to_datetime(news.date)
    news["day"] = news.date.dt.normalize()
    days = news[news.day.isin(f.index)].groupby("day").title.apply(lambda t: " ".join(t.astype(str)))
    vec = TfidfVectorizer(max_features=3000, stop_words="english").fit(days[days.index < TRAIN_END])
    day_mat = vec.transform(days)
    day_pos = np.array([pos[d] for d in days.index])

    news_pred, covered = [], 0
    for i, p in zip(val, price):
        t = f.index[i]
        recent = news[(news.date < t) & (news.date >= t - pd.Timedelta(days=7))]
        ok = day_pos + H <= i  # outcome finished by t
        if recent.empty or not ok.any():
            news_pred.append(p)  # no fresh news: fall back to the price model, like the dashboard
            continue
        covered += 1
        query = " ".join(recent.sort_values("date").tail(10).title.astype(str))
        sims = cosine_similarity(vec.transform([query]), day_mat[ok]).ravel()
        top = np.argsort(sims)[::-1][:TOP_K]
        outcomes = f.target5.iloc[day_pos[ok][top]].to_numpy()
        news_pred.append(np.average(outcomes, weights=np.maximum(sims[top], 0.001)))

    news_pred, actual = np.array(news_pred), f.target5.iloc[val].to_numpy()
    rows = {
        "Always up (+1%)": score(np.full(len(val), 0.01), actual),
        "Price model": score(price, actual),
        "News only": score(news_pred, actual),
        "70/30 blend (dashboard)": score((1 - NEWS_WEIGHT) * price + NEWS_WEIGHT * news_pred, actual),
    }
    return rows, len(val), covered


if __name__ == "__main__":
    for ticker in [t.upper() for t in sys.argv[1:]] or TICKERS:
        rows, n, covered = evaluate(ticker)
        print(f"\n{ticker}: {n} validation forecasts, {covered} with news in the prior 7 days")
        print(pd.DataFrame(rows).T.to_string(float_format=lambda x: f"{x:.3f}"))
