import io, json, re, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity

CACHE = Path("data/cache")
CACHE.mkdir(parents=True, exist_ok=True)
TICKERS = ["NVDA", "TSLA", "AAPL", "SPY"]
NAMES = {"NVDA": "Nvidia", "TSLA": "Tesla", "AAPL": "Apple", "SPY": "S&P 500"}
NEWS_COLS = ["date", "title", "url", "source"]
FEATURES = ["ret1", "mom5", "mom20", "vol20", "vol_ratio", "spy5"]


def mentions(ticker):
    """Regex for a headline about this ticker: the ticker in capitals, or the company name in any case."""
    return rf"(?<![A-Za-z0-9]){ticker}(?![A-Za-z0-9])|(?i:\b{re.escape(NAMES[ticker])}\b)"


def prices(ticker, period="max"):
    p = CACHE / f"{ticker}_{period}.parquet"
    if p.exists() and time.time() - p.stat().st_mtime < 3600:
        return pd.read_parquet(p)
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No price data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    df.to_parquet(p)
    return df


def gdelt(ticker, days=90):
    p = CACHE / f"news_{ticker}_{days}.parquet"
    if p.exists() and time.time() - p.stat().st_mtime < 1800:
        return pd.read_parquet(p)
    params = {
        "query": f'({ticker} OR "{NAMES[ticker]}") sourcelang:eng',
        "mode": "ArtList",
        "format": "json",
        "maxrecords": 250,
        "timespan": f"{days}d",
        "sort": "HybridRel",
    }
    for attempt in range(3):
        try:
            r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=25)
            r.raise_for_status()  # includes 429 rate limiting
            articles = r.json().get("articles", [])
            break
        except (requests.RequestException, ValueError, AttributeError):
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
    else:
        # GDELT unavailable: fall back to the last cached copy, however old.
        return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=NEWS_COLS)
    df = pd.DataFrame(
        [
            {
                "date": pd.to_datetime(x.get("seendate"), format="%Y%m%dT%H%M%SZ", errors="coerce"),
                "title": x.get("title", ""),
                "url": x.get("url", ""),
                "source": x.get("domain", "GDELT"),
            }
            for x in articles
        ],
        columns=NEWS_COLS,
    ).dropna(subset=["date"]).drop_duplicates("title")
    df.to_parquet(p)
    return df


def _records(name, raw):
    """Article dicts from one CSV/JSON/JSONL file's bytes."""
    suffix = Path(name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(io.BytesIO(raw), low_memory=False).to_dict("records")
    if suffix in (".jsonl", ".ndjson"):
        return [json.loads(line) for line in raw.decode("utf-8", "ignore").splitlines() if line.strip()]
    if suffix == ".json":
        v = json.loads(raw.decode("utf-8", "ignore"))
        return v if isinstance(v, list) else v.get("posts", v.get("articles", [v]))
    return []


def webhose_news(ticker):
    # Loose CSV/JSON/JSONL files, or Webhose's weekly ZIP releases containing them.
    rows = []
    for f in Path("data/webhose").glob("**/*"):
        try:
            if f.suffix.lower() == ".zip":
                with zipfile.ZipFile(f) as z:
                    records = [x for m in z.namelist() for x in _records(m, z.read(m))]
            elif f.is_file():
                records = _records(f.name, f.read_bytes())
            else:
                continue
            for x in records:
                if not isinstance(x, dict):
                    continue
                thread = x.get("thread") if isinstance(x.get("thread"), dict) else {}
                title = str(x.get("title") or x.get("headline") or x.get("name") or "")
                body = str(x.get("text") or x.get("description") or x.get("snippet") or "")
                symbols = x.get("symbols") or x.get("tickers") or []
                if isinstance(symbols, str):
                    symbols = [symbols]
                if ticker not in [str(s).upper() for s in symbols] and not re.search(mentions(ticker), title + " " + body):
                    continue
                date = x.get("published") or x.get("published_at") or x.get("date") or thread.get("published")
                dt = pd.to_datetime(date, errors="coerce", utc=True)
                if pd.notna(dt):
                    rows.append(
                        {
                            "date": dt.tz_localize(None),
                            "title": title,
                            "url": x.get("url") or thread.get("url", ""),
                            "source": x.get("site") or x.get("domain") or thread.get("site") or f.name,
                        }
                    )
        except Exception:
            continue  # skip unreadable files
    return pd.DataFrame(rows, columns=NEWS_COLS).drop_duplicates("title")


def dataset_news(ticker):
    # CSVs written by scripts/download_datasets.py: date, title, url, source, ticker
    files = list(Path("data/datasets").glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=NEWS_COLS)
    d = pd.concat(map(pd.read_csv, files), ignore_index=True)
    d = d[d.ticker == ticker].fillna("")
    d["date"] = pd.to_datetime(d.date, errors="coerce")
    return d.dropna(subset=["date"])[NEWS_COLS]


def all_news(ticker):
    news = pd.concat([gdelt(ticker), webhose_news(ticker), dataset_news(ticker)], ignore_index=True)
    news["date"] = pd.to_datetime(news.date)
    return news.drop_duplicates("title")


def market_frame(ticker):
    """Daily features, plus target5 = next 5-day return (NaN for the latest 5 days)."""
    d, spy = prices(ticker), prices("SPY")
    c, v = d["Close"], d["Volume"]
    r = c.pct_change()
    out = pd.DataFrame(
        {
            "ret1": r,
            "mom5": c.pct_change(5),
            "mom20": c.pct_change(20),
            "vol20": r.rolling(20).std() * np.sqrt(252),
            "vol_ratio": v / v.rolling(20).mean(),
            "spy5": spy["Close"].pct_change().reindex(d.index).fillna(0).rolling(5).sum(),
            "target5": c.shift(-5) / c - 1,
        }
    )
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)


def forecast_bundle(frame, news):
    known = frame.dropna(subset=["target5"])  # days whose 5-day outcome is known
    model = Ridge(alpha=1).fit(known[FEATURES], known.target5)
    base = float(model.predict(frame[FEATURES].iloc[[-1]])[0])  # predict from today

    # Today's story = the 10 newest headlines; compare it to each past day's headlines.
    news = news.assign(day=news.date.dt.normalize())
    latest = news.sort_values("date", ascending=False).head(10)
    past = news[news.day.isin(known.index) & ~news.day.isin(latest.day)]
    corpus = past.groupby("day").title.apply(lambda t: " · ".join(t.astype(str)))
    match = []
    if len(corpus):
        query = " ".join(latest.title.astype(str))
        mat = TfidfVectorizer(max_features=3000, stop_words="english").fit_transform([*corpus, query])
        sims = cosine_similarity(mat[-1], mat[:-1]).ravel()
        for i in np.argsort(sims)[::-1][:15]:
            day = corpus.index[i]
            match.append(
                {
                    "date": day.strftime("%Y-%m-%d"),
                    "headlines": corpus.iloc[i][:500],
                    "similarity": float(sims[i]),
                    "actual_5d": float(known.target5[day]),
                }
            )
    weighted = (
        np.average([m["actual_5d"] for m in match], weights=[max(m["similarity"], 0.001) for m in match])
        if match
        else base
    )
    return {
        "prediction": 0.7 * base + 0.3 * weighted,
        "market_baseline": base,
        "momentum_5d": float(frame.mom5.iloc[-1]),
        "matches": match,
        "positive_match_rate": float(np.mean([m["actual_5d"] > 0 for m in match])) if match else None,
        "news_count": len(news),
    }
