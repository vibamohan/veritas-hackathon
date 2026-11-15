from __future__ import annotations
import re, time, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

CACHE = Path("data/cache")
CACHE.mkdir(parents=True, exist_ok=True)
TICKERS = ["NVDA", "TSLA", "AAPL", "SPY"]


def prices(ticker, period="5y"):
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
    q = requests.get(
        "https://api.gdeltproject.org/api/v2/doc/doc",
        params={
            "query": f'"{ticker}" sourcelang:eng',
            "mode": "ArtList",
            "format": "json",
            "maxrecords": 250,
            "timespan": f"{days}d",
            "sort": "HybridRel",
        },
        timeout=25,
    )
    q.raise_for_status()
    rows = q.json().get("articles", [])
    df = pd.DataFrame(
        [
            {
                "date": pd.to_datetime(
                    x.get("seendate"), format="%Y%m%dT%H%M%SZ", errors="coerce"
                ),
                "title": x.get("title", ""),
                "url": x.get("url", ""),
                "source": x.get("domain", "GDELT"),
            }
            for x in rows
        ]
    )
    if not df.empty:
        df = df.dropna(subset=["date"]).drop_duplicates("title")
    df.to_parquet(p)
    return df


def webhose_news(ticker):
    root = Path("data/webhose")
    rows = []
    for f in root.glob("**/*"):
        if not f.is_file():
            continue
        try:
            if f.suffix.lower() == ".zip":
                # Webhose's financial-news repository distributes each weekly
                # release as a ZIP containing JSON/JSONL article records.
                with zipfile.ZipFile(f) as archive:
                    records = []
                    for member in archive.namelist():
                        if member.endswith("/"):
                            continue
                        suffix = Path(member).suffix.lower()
                        with archive.open(member) as stream:
                            if suffix == ".csv":
                                records.extend(pd.read_csv(stream, low_memory=False).to_dict("records"))
                            elif suffix in (".jsonl", ".ndjson"):
                                records.extend(json.loads(line) for line in stream.read().decode("utf-8", "ignore").splitlines() if line.strip())
                            elif suffix == ".json":
                                value = json.load(stream)
                                records.extend(value if isinstance(value, list) else value.get("posts", value.get("articles", [value])))
                    d = records
            elif f.suffix.lower() == ".csv":
                d = pd.read_csv(f, low_memory=False).to_dict("records")
            elif f.suffix.lower() in (".jsonl", ".ndjson"):
                d = [
                    __import__("json").loads(x)
                    for x in f.read_text(errors="ignore").splitlines()
                    if x.strip()
                ]
            elif f.suffix.lower() == ".json":
                v = __import__("json").loads(f.read_text(errors="ignore"))
                d = v if isinstance(v, list) else v.get("posts", v.get("articles", []))
            else:
                continue
            for x in d:
                if not isinstance(x, dict):
                    continue
                thread = x.get("thread") if isinstance(x.get("thread"), dict) else {}
                title = x.get("title") or x.get("headline") or x.get("name") or ""
                body = x.get("text") or x.get("description") or x.get("snippet") or ""
                date = x.get("published") or x.get("published_at") or x.get("date") or thread.get("published")
                blob = (str(title) + " " + str(body)).lower()
                symbols = x.get("symbols") or x.get("tickers") or []
                if isinstance(symbols, str):
                    symbols = [symbols]
                symbol_match = any(str(s).upper() == ticker.upper() for s in symbols)
                # Match ticker tokens, not substrings (e.g. AAPL in "Pineapple").
                if not symbol_match and not re.search(rf"(?<![A-Z0-9]){re.escape(ticker)}(?![A-Z0-9])", blob, re.I):
                    continue
                dt = pd.to_datetime(date, errors="coerce", utc=True)
                if pd.isna(dt):
                    continue
                rows.append(
                    {
                        "date": dt.tz_localize(None),
                        "title": str(title),
                        "url": x.get("url") or thread.get("url", ""),
                        "source": x.get("site") or x.get("domain") or thread.get("site") or f.name,
                        "sentiment": x.get("sentiment") or (re.search(r"_(positive|negative)_", f.name, re.I).group(1).lower() if re.search(r"_(positive|negative)_", f.name, re.I) else "unknown"),
                        "text": str(body),
                    }
                )
        except Exception:
            continue
    return (
        pd.DataFrame(rows).drop_duplicates("title")
        if rows
        else pd.DataFrame(columns=["date", "title", "url", "source", "sentiment", "text"])
    )


def kaggle_news(ticker):
    root = Path("data/kaggle")
    frames = []
    for f in root.glob("*.csv"):
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        cols = {c.lower(): c for c in d.columns}
        date = next(
            (cols[x] for x in cols if any(k in x for k in ["date", "time"])), None
        )
        title = next(
            (
                cols[x]
                for x in cols
                if any(k in x for k in ["title", "headline", "news", "text"])
            ),
            None,
        )
        sym = next(
            (cols[x] for x in cols if any(k in x for k in ["ticker", "symbol"])), None
        )
        if date and title:
            z = pd.DataFrame(
                {
                    "date": pd.to_datetime(d[date], errors="coerce"),
                    "title": d[title].astype(str),
                    "url": "",
                    "source": f.name,
                }
            )
            if sym:
                z = z[
                    d[sym]
                    .astype(str)
                    .str.contains(rf"\b{ticker}\b", case=False, regex=True, na=False)
                ]
            frames.append(z.dropna(subset=["date"]))
    return (
        pd.concat(frames, ignore_index=True).drop_duplicates("title")
        if frames
        else pd.DataFrame(columns=["date", "title", "url", "source"])
    )


def market_frame(ticker):
    d = prices(ticker)
    spy = prices("SPY")
    c = d["Close"].squeeze()
    r = c.pct_change()
    s = spy["Close"].squeeze().pct_change().reindex(d.index).fillna(0)
    out = pd.DataFrame(index=d.index)
    out["ret1"] = r
    out["mom5"] = c.pct_change(5)
    out["mom20"] = c.pct_change(20)
    out["vol20"] = r.rolling(20).std() * np.sqrt(252)
    out["vol_ratio"] = d["Volume"].squeeze() / d["Volume"].squeeze().rolling(20).mean()
    out["spy5"] = s.rolling(5).sum()
    out["target5"] = c.shift(-5) / c - 1
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def forecast_bundle(ticker, frame, news):
    cols = ["ret1", "mom5", "mom20", "vol20", "vol_ratio", "spy5"]
    X = frame[cols]
    y = frame.target5
    train = frame.iloc[:-5]
    base_model = Ridge(alpha=1).fit(train[cols], train.target5)
    base = float(base_model.predict(X.iloc[[-1]])[0])
    texts = []
    for _, r in news.dropna(subset=["date"]).iterrows():
        day = pd.to_datetime(r.date).normalize()
        if day in frame.index:
            texts.append((day, str(r.title)))
    corpus = {}
    for d, txt in texts:
        corpus.setdefault(d, []).append(txt)
    dates = list(corpus)
    match = []
    if dates:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        docs = [" ".join(corpus[d]) for d in dates]
        query = " ".join(news.head(10).title.astype(str))
        mat = TfidfVectorizer(max_features=3000, stop_words="english").fit_transform(
            docs + [query]
        )
        sims = cosine_similarity(mat[-1], mat[:-1]).ravel()
        for d, s in zip(dates, sims):
            if d in frame.index and pd.notna(frame.loc[d, "target5"]):
                match.append(
                    {
                        "date": d.strftime("%Y-%m-%d"),
                        "headlines": " · ".join(corpus[d])[:500],
                        "similarity": float(s),
                        "actual_5d": float(frame.loc[d, "target5"]),
                    }
                )
    match = sorted(match, key=lambda x: x["similarity"], reverse=True)[:15]
    weighted = (
        np.average(
            [m["actual_5d"] for m in match],
            weights=[max(m["similarity"], 0.001) for m in match],
        )
        if match
        else base
    )
    return {
        "prediction": 0.7 * base + 0.3 * weighted,
        "market_baseline": base,
        "matches": match,
        "positive_match_rate": float(np.mean([m["actual_5d"] > 0 for m in match]))
        if match
        else None,
        "news_count": len(news),
    }
