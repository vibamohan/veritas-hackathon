"""Data loading and the dashboard's direction model.

Sections:
  1. Settings shared by the app and the scripts
  2. Prices (yfinance)
  3. News: live GDELT headlines, Webhose archives, and the committed historical datasets
  4. The experimental 5-day direction forecast shown on the dashboard

The volatility ("turbulence") model lives in scripts/volatility.py.
"""
import io
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------

CACHE = Path("data/cache")  # downloaded prices/news are cached here (git-ignored)
CACHE.mkdir(parents=True, exist_ok=True)

TICKERS = ["NVDA", "TSLA", "AAPL", "SPY"]
NAMES = {"NVDA": "Nvidia", "TSLA": "Tesla", "AAPL": "Apple", "SPY": "S&P 500"}  # used to match headlines
NEWS_COLS = ["date", "title", "url", "source"]  # every news loader returns these columns

# Price features for the direction model: 1-day return, 5/20-day momentum,
# 20-day volatility, volume vs its 20-day average, and SPY's 5-day return.
FEATURES = ["ret1", "mom5", "mom20", "vol20", "vol_ratio", "spy5"]
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)  # regularisation strengths tried by the experiment scripts


def ridge_model(alpha=1.0):
    """Ridge regression with standardised inputs (the features have very different scales)."""
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def mentions(ticker):
    """Regex that matches a headline about `ticker`: the ticker in capitals (not inside a longer word,
    so "AAPL" doesn't match "AAPLX"), or the company name in any case ("nvidia", "NVIDIA")."""
    return rf"(?<![A-Za-z0-9]){ticker}(?![A-Za-z0-9])|(?i:\b{re.escape(NAMES[ticker])}\b)"


# ---------------------------------------------------------------------------
# 2. Prices
# ---------------------------------------------------------------------------

def prices(ticker, period="max"):
    """Daily OHLCV prices from yfinance, cached for an hour. Falls back to an older cached copy
    when Yahoo is unavailable."""
    path = CACHE / f"{ticker}_{period}.parquet"
    if path.exists() and time.time() - path.stat().st_mtime < 3600:
        return pd.read_parquet(path)
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        if path.exists():
            return pd.read_parquet(path)  # stale, but better than failing
        raise RuntimeError(f"No price data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):  # yfinance sometimes returns (field, ticker) columns
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    df.to_parquet(path)
    return df


# ---------------------------------------------------------------------------
# 3. News
# ---------------------------------------------------------------------------

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_gdelt_failed_at = {}  # ticker -> time of the last failed GDELT request


def _fetch_gdelt_articles(params):
    """Ask GDELT for articles, retrying network errors. Returns None if it fails.
    A 429 ("too many requests") is not retried: retrying only extends the rate limit."""
    for attempt in range(3):
        try:
            response = requests.get(GDELT_URL, params=params, timeout=15)
            response.raise_for_status()
            return response.json().get("articles", [])
        except (requests.RequestException, ValueError, AttributeError) as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            if status == 429 or attempt == 2:
                return None
            time.sleep(2 ** (attempt + 1))  # wait 2s, then 4s
    return None


def gdelt(ticker, days=90):
    """Live headlines from the last `days` days via the free GDELT API.
    Cached for 30 minutes. If GDELT fails, the last cached copy is used and GDELT is not asked
    again for 10 minutes, so a rate limit can't slow every page load."""
    path = CACHE / f"news_{ticker}_{days}.parquet"
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=NEWS_COLS)
    cache_is_fresh = path.exists() and time.time() - path.stat().st_mtime < 1800
    failed_recently = time.time() - _gdelt_failed_at.get(ticker, 0) < 600
    if cache_is_fresh or failed_recently:
        return cached

    articles = _fetch_gdelt_articles({
        "query": f'({ticker} OR "{NAMES[ticker]}") sourcelang:eng',
        "mode": "ArtList",
        "format": "json",
        "maxrecords": 250,
        "timespan": f"{days}d",
        "sort": "HybridRel",
    })
    if articles is None:
        _gdelt_failed_at[ticker] = time.time()
        return cached

    rows = [
        {
            "date": pd.to_datetime(a.get("seendate"), format="%Y%m%dT%H%M%SZ", errors="coerce"),
            "title": a.get("title", ""),
            "url": a.get("url", ""),
            "source": a.get("domain", "GDELT"),
        }
        for a in articles
    ]
    df = pd.DataFrame(rows, columns=NEWS_COLS).dropna(subset=["date"]).drop_duplicates("title")
    df.to_parquet(path)
    return df


def _read_records(filename, raw):
    """Turn one CSV / JSON / JSONL file (as bytes) into a list of article dicts."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(io.BytesIO(raw), low_memory=False).to_dict("records")
    text = raw.decode("utf-8", "ignore")
    if suffix in (".jsonl", ".ndjson"):  # one JSON object per line
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if suffix == ".json":
        value = json.loads(text)
        return value if isinstance(value, list) else value.get("posts", value.get("articles", [value]))
    return []


def _webhose_row(article, ticker, source_file):
    """Convert one Webhose article to a news row, or None if it isn't about `ticker` or has no date.
    Field names vary between releases, so several alternatives are tried."""
    if not isinstance(article, dict):
        return None
    thread = article.get("thread") if isinstance(article.get("thread"), dict) else {}
    title = str(article.get("title") or article.get("headline") or article.get("name") or "")
    body = str(article.get("text") or article.get("description") or article.get("snippet") or "")

    symbols = article.get("symbols") or article.get("tickers") or []
    if isinstance(symbols, str):
        symbols = [symbols]
    tagged = ticker in [str(s).upper() for s in symbols]
    if not tagged and not re.search(mentions(ticker), title + " " + body):
        return None

    date = article.get("published") or article.get("published_at") or article.get("date") or thread.get("published")
    when = pd.to_datetime(date, errors="coerce", utc=True)
    if pd.isna(when):
        return None
    return {
        "date": when.tz_localize(None),
        "title": title,
        "url": article.get("url") or thread.get("url", ""),
        "source": article.get("site") or article.get("domain") or thread.get("site") or source_file,
    }


def webhose_news(ticker):
    """Optional Webhose articles from data/webhose/: loose CSV/JSON/JSONL files or weekly ZIP releases."""
    rows = []
    for path in Path("data/webhose").glob("**/*"):
        try:
            if path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as archive:
                    articles = [a for name in archive.namelist() for a in _read_records(name, archive.read(name))]
            elif path.is_file():
                articles = _read_records(path.name, path.read_bytes())
            else:
                continue
            rows += [row for a in articles if (row := _webhose_row(a, ticker, path.name))]
        except Exception:
            continue  # skip unreadable files instead of failing the whole dashboard
    return pd.DataFrame(rows, columns=NEWS_COLS).drop_duplicates("title")


def dataset_news(ticker):
    """Historical headlines from the committed CSVs in data/datasets/
    (built by scripts/download_datasets.py: date, title, url, source, ticker)."""
    files = list(Path("data/datasets").glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=NEWS_COLS)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df.ticker == ticker].fillna("")
    df["date"] = pd.to_datetime(df.date, errors="coerce")
    return df.dropna(subset=["date"])[NEWS_COLS]


def all_news(ticker):
    """Every headline we have for `ticker`: live GDELT + Webhose + historical datasets."""
    news = pd.concat([gdelt(ticker), webhose_news(ticker), dataset_news(ticker)], ignore_index=True)
    news["date"] = pd.to_datetime(news.date)
    return news.drop_duplicates("title")


# ---------------------------------------------------------------------------
# 4. The experimental 5-day direction forecast
# (testing showed it does not beat "always up"; see docs/EXPERIMENTS.md)
# ---------------------------------------------------------------------------

def market_frame(ticker):
    """One row per trading day: the price FEATURES, plus target5 = the return over the next 5 days.
    target5 is NaN for the latest 5 days because their outcome isn't known yet."""
    stock, spy = prices(ticker), prices("SPY")
    close, volume = stock["Close"], stock["Volume"]
    daily_return = close.pct_change()
    frame = pd.DataFrame({
        "ret1": daily_return,
        "mom5": close.pct_change(5),
        "mom20": close.pct_change(20),
        "vol20": daily_return.rolling(20).std() * np.sqrt(252),  # annualised volatility
        "vol_ratio": volume / volume.rolling(20).mean(),
        "spy5": spy["Close"].pct_change().reindex(stock.index).fillna(0).rolling(5).sum(),
        "target5": close.shift(-5) / close - 1,
    })
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)


def _join_headlines(titles):
    return " · ".join(titles.astype(str))


def _similar_news_days(news, known, top_k=15):
    """Find the past days whose headlines are most similar (TF-IDF cosine) to today's 10 newest
    headlines, and what the stock did over the following 5 days."""
    news = news.assign(day=news.date.dt.normalize())
    latest = news.sort_values("date", ascending=False).head(10)
    # Candidate days: outcome already known, and not one of the days the query headlines come from.
    past = news[news.day.isin(known.index) & ~news.day.isin(latest.day)]
    day_docs = past.groupby("day").title.apply(_join_headlines)  # one text per day
    if day_docs.empty:
        return []

    query = " ".join(latest.title.astype(str))
    vectors = TfidfVectorizer(max_features=3000, stop_words="english").fit_transform([*day_docs, query])
    similarity = cosine_similarity(vectors[-1], vectors[:-1]).ravel()
    matches = []
    for i in np.argsort(similarity)[::-1][:top_k]:  # most similar first
        day = day_docs.index[i]
        matches.append({
            "date": day.strftime("%Y-%m-%d"),
            "headlines": day_docs.iloc[i][:500],
            "similarity": float(similarity[i]),
            "actual_5d": float(known.target5[day]),
        })
    return matches


def forecast_bundle(frame, news):
    """Dashboard direction forecast: 70% price model + 30% similarity-weighted outcome of the
    most similar past news days."""
    known = frame.dropna(subset=["target5"])  # days whose 5-day outcome is known
    price_model = ridge_model().fit(known[FEATURES], known.target5)
    base = float(price_model.predict(frame[FEATURES].iloc[[-1]])[0])  # forecast from today's features

    matches = _similar_news_days(news, known)
    if matches:
        outcomes = [m["actual_5d"] for m in matches]
        weights = [max(m["similarity"], 0.001) for m in matches]
        news_estimate = np.average(outcomes, weights=weights)
    else:
        news_estimate = base

    return {
        "prediction": 0.7 * base + 0.3 * news_estimate,
        "market_baseline": base,
        "momentum_5d": float(frame.mom5.iloc[-1]),
        "matches": matches,
        "positive_match_rate": float(np.mean([m["actual_5d"] > 0 for m in matches])) if matches else None,
        "news_count": len(news),
    }
