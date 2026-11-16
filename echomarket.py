"""data loading and the dashboard's direction model.

sections:
  1. settings shared by the app and the scripts
  2. prices (yfinance)
  3. news: live GDELT headlines, webhose archives, and the committed historical datasets
  4. the experimental 5-day direction forecast shown on the dashboard

the volatility ("turbulence") model lives in scripts/volatility.py.
"""
import io
import json
import re
import threading
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
# 1. settings
# ---------------------------------------------------------------------------

CACHE = Path("data/cache")  # downloaded prices/news are cached here (git-ignored)
CACHE.mkdir(parents=True, exist_ok=True)

# the four stocks the app is built and tested around (any other symbol can still be looked up).
TICKERS = ["NVDA", "TSLA", "AAPL", "SPY"]
NAMES = {
    "NVDA": "Nvidia",
    "TSLA": "Tesla",
    "AAPL": "Apple",
    "SPY": "S&P 500",
    "DIS": "Walt Disney",
}  # used to match headlines and broaden live news searches
# extra search words for some tickers (people write "disney", not "walt disney").
NEWS_ALIASES = {"DIS": ["Disney"]}
NEWS_COLS = ["date", "title", "url", "source"]  # every news loader returns these columns
# company names looked up on yahoo, saved to disk so we only ask once a week.
COMPANY_NAME_CACHE = CACHE / "company_names.json"
_company_name_cache = None

# price features for the direction model: 1-day return, 5/20-day momentum,
# 20-day volatility, volume vs its 20-day average, and SPY's 5-day return.
FEATURES = ["ret1", "mom5", "mom20", "vol20", "vol_ratio", "spy5"]
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)  # regularisation strengths tried by the experiment scripts


def save_parquet(df, path):
    """write via a temporary file, then rename: readers (and other threads) never see a half-written file."""
    # write to a scratch file first, then swap it in with one quick rename.
    temporary = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
    df.to_parquet(temporary)
    temporary.replace(path)


def ridge_model(alpha=1.0):
    """ridge regression with standardised inputs (the features have very different scales)."""
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def mentions(ticker):
    """regex for the ticker or known company name appearing in a headline."""
    # every way a headline could name this stock: the ticker, company names and aliases.
    terms = [ticker, *company_names(ticker), *NEWS_ALIASES.get(ticker, [])]
    # the ticker must match exactly in capitals (so "AAPL" isn't found inside "AAPLX"),
    # while company names match in any case ("nvidia" or "NVIDIA").
    return "|".join(
        rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])"
        if term == ticker
        else rf"(?i:\b{re.escape(term)}\b)"
        for term in terms
    )


def company_names(ticker):
    """return known and yahoo finance company names for any valid ticker.

    yahoo lookups are cached on disk for seven days. if metadata is unavailable, the ticker
    remains a usable search term and the built-in names for the default symbols still apply.
    """
    # load the saved name lookups from disk the first time this runs.
    global _company_name_cache
    if _company_name_cache is None:
        try:
            _company_name_cache = json.loads(COMPANY_NAME_CACHE.read_text())
        except (OSError, ValueError):
            _company_name_cache = {}

    # start with the names we already know: our built-in list plus anything saved.
    ticker = ticker.upper()
    cached = _company_name_cache.get(ticker, {})
    names = [NAMES.get(ticker)]
    names.extend(cached.get("names", []))
    # for a ticker we don't know, ask yahoo for its company name (at most once a week).
    if ticker not in NAMES and (
        not cached or time.time() - cached.get("updated_at", 0) >= 7 * 24 * 3600
    ):
        try:
            info = yf.Ticker(ticker).get_info()
            fetched = [info.get("longName"), info.get("shortName")]
            fetched = list(dict.fromkeys(str(name).strip() for name in fetched if name and str(name).strip()))
            names.extend(fetched)
            _company_name_cache[ticker] = {"names": fetched, "updated_at": time.time()}
            COMPANY_NAME_CACHE.write_text(json.dumps(_company_name_cache))
        except Exception:
            pass  # news search can still use the ticker and any cached/static names.

    # remove blanks and duplicates but keep the order.
    return list(dict.fromkeys(name for name in names if name))


def valid_ticker(ticker):
    """accept common yahoo finance symbols, including class-share dots and hyphens."""
    # 1 to 15 characters: capital letters, digits, dots and dashes (like BRK-B).
    return bool(re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,14}", ticker))


# ---------------------------------------------------------------------------
# 2. prices
# ---------------------------------------------------------------------------

def prices(ticker, period="max"):
    """daily OHLCV prices from yfinance, cached for an hour. falls back to an older cached copy
    when yahoo is unavailable."""
    # reuse the saved copy if it's less than an hour old.
    path = CACHE / f"{ticker}_{period}.parquet"
    if path.exists() and time.time() - path.stat().st_mtime < 3600:
        return pd.read_parquet(path)
    # otherwise download the full price history from yahoo.
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    except Exception:
        df = pd.DataFrame()
    # yahoo failed or sent nothing back: use the old saved copy if there is one.
    if df.empty:
        if path.exists():
            return pd.read_parquet(path)  # stale, but better than failing
        raise RuntimeError(f"No price data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):  # yfinance sometimes returns (field, ticker) columns
        df.columns = df.columns.get_level_values(0)
    # clean up the dates (remove time zones) and save the result for next time.
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    save_parquet(df, path)
    return df


# ---------------------------------------------------------------------------
# 3. news
# ---------------------------------------------------------------------------

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_gdelt_failed_at = {}  # ticker -> time of the last failed GDELT request


def _fetch_gdelt_articles(params, attempts=3):
    """ask GDELT for articles, retrying network errors. returns none if it fails.
    a 429 ("too many requests") is not retried: retrying only extends the rate limit."""
    # try a few times, because the free GDELT service often times out.
    for attempt in range(attempts):
        try:
            response = requests.get(GDELT_URL, params=params, timeout=15)
            response.raise_for_status()
            return response.json().get("articles", [])
        except (requests.RequestException, ValueError, AttributeError) as error:
            # 429 means "slow down": give up right away instead of making it worse.
            status = getattr(getattr(error, "response", None), "status_code", None)
            if status == 429 or attempt == attempts - 1:
                return None
            time.sleep(2 ** (attempt + 1))  # wait 2s, then 4s
    return None


_gdelt_refreshing = set()  # tickers with a background refresh in progress
_gdelt_lock = threading.Lock()


def gdelt(ticker, days=90):
    """live headlines from the last `days` days via the free GDELT API.

    GDELT is slow (often 10+ seconds) and rate-limited, so the dashboard never waits for it when
    any cached copy exists: the cached headlines are returned at once and, if older than 30 minutes,
    refreshed in a background thread for the next request. only the very first load waits."""
    path = CACHE / f"news_{ticker}_{days}.parquet"
    # no saved headlines yet: this one time we have to wait for GDELT.
    if not path.exists():
        return _refresh_gdelt(ticker, days)
    # saved headlines are over 30 minutes old: refresh them in the background,
    # and only start one refresh per ticker at a time.
    if time.time() - path.stat().st_mtime > 1800:
        with _gdelt_lock:
            start = ticker not in _gdelt_refreshing
            _gdelt_refreshing.add(ticker)
        if start:
            threading.Thread(target=_refresh_gdelt, args=(ticker, days, True), daemon=True).start()
    # answer right away with the saved headlines.
    return pd.read_parquet(path)


def _refresh_gdelt(ticker, days, background=False):
    """download headlines from GDELT and update the cache. if GDELT fails, keep the old copy
    and don't ask again for 10 minutes."""
    path = CACHE / f"news_{ticker}_{days}.parquet"
    # whatever headlines we already have saved (or an empty table).
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=NEWS_COLS)
    try:
        # GDELT failed less than 10 minutes ago: don't ask again yet.
        if time.time() - _gdelt_failed_at.get(ticker, 0) < 600:
            return cached
        # someone is waiting in the foreground: try once. in the background: retry.
        return _download_gdelt(ticker, days, path, cached, attempts=3 if background else 1)
    # mark this ticker's background refresh as done, even if it failed.
    finally:
        if background:
            with _gdelt_lock:
                _gdelt_refreshing.discard(ticker)


def _download_gdelt(ticker, days, path, cached, attempts):
    """ask gdelt for recent articles about the ticker, save them to the cache, and return them."""
    # search for the ticker or any known company name / alias (works for custom tickers too).
    search_terms = [ticker, *(f'"{name}"' for name in company_names(ticker)), *NEWS_ALIASES.get(ticker, [])]
    query = f"({' OR '.join(search_terms)}) sourcelang:eng"
    # ask for up to 250 english articles from the last `days` days.
    articles = _fetch_gdelt_articles({
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": 250,
        "timespan": f"{days}d",
        "sort": "HybridRel",
    }, attempts)
    # the request failed: remember when, and keep the old saved headlines.
    if articles is None:
        _gdelt_failed_at[ticker] = time.time()
        return cached

    # keep just the date, headline, link and website of each article.
    rows = [
        {
            "date": pd.to_datetime(a.get("seendate"), format="%Y%m%dT%H%M%SZ", errors="coerce"),
            "title": a.get("title", ""),
            "url": a.get("url", ""),
            "source": a.get("domain", "GDELT"),
        }
        for a in articles
    ]
    # drop articles without a date and repeated headlines, then save them.
    df = pd.DataFrame(rows, columns=NEWS_COLS).dropna(subset=["date"]).drop_duplicates("title")
    save_parquet(df, path)
    return df


def _read_records(filename, raw):
    """turn one CSV / JSON / JSONL file (as bytes) into a list of article dicts."""
    # pick the right reader based on the file type.
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
    """convert one webhose article to a news row, or none if it isn't about `ticker` or has no date.
    field names vary between releases, so several alternatives are tried."""
    if not isinstance(article, dict):
        return None
    # webhose keeps some fields inside a nested "thread" object.
    thread = article.get("thread") if isinstance(article.get("thread"), dict) else {}
    title = str(article.get("title") or article.get("headline") or article.get("name") or "")
    body = str(article.get("text") or article.get("description") or article.get("snippet") or "")

    # keep the article only if it's tagged with the ticker or mentions the company.
    symbols = article.get("symbols") or article.get("tickers") or []
    if isinstance(symbols, str):
        symbols = [symbols]
    tagged = ticker in [str(s).upper() for s in symbols]
    if not tagged and not re.search(mentions(ticker), title + " " + body):
        return None

    # find the publish date; skip articles that don't have one.
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
    """optional webhose articles from data/webhose/: loose CSV/JSON/JSONL files or weekly ZIP releases."""
    rows = []
    # go through every file in data/webhose (this folder is optional and usually empty).
    for path in Path("data/webhose").glob("**/*"):
        try:
            if path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as archive:
                    articles = [a for name in archive.namelist() for a in _read_records(name, archive.read(name))]
            elif path.is_file():
                articles = _read_records(path.name, path.read_bytes())
            else:
                continue
            # turn each article into a row, keeping only the ones about this ticker.
            rows += [row for a in articles if (row := _webhose_row(a, ticker, path.name))]
        except Exception:
            continue  # skip unreadable files instead of failing the whole dashboard
    return pd.DataFrame(rows, columns=NEWS_COLS).drop_duplicates("title")


def dataset_news(ticker):
    """historical headlines from the committed CSVs in data/datasets/
    (built by scripts/download_datasets.py: date, title, url, source, ticker)."""
    # these CSVs were already filtered to our tickers, so they're small and quick to read.
    files = list(Path("data/datasets").glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=NEWS_COLS)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # keep this ticker's rows and turn the date text into real dates.
    df = df[df.ticker == ticker].fillna("")
    df["date"] = pd.to_datetime(df.date, errors="coerce")
    return df.dropna(subset=["date"])[NEWS_COLS]


def all_news(ticker):
    """every headline we have for `ticker`: live GDELT + webhose + historical datasets."""
    # stack all three sources together and remove repeated headlines.
    news = pd.concat([gdelt(ticker), webhose_news(ticker), dataset_news(ticker)], ignore_index=True)
    news["date"] = pd.to_datetime(news.date)
    return news.drop_duplicates("title")


# ---------------------------------------------------------------------------
# 4. the experimental 5-day direction forecast
# (testing showed it does not beat "always up"; see docs/EXPERIMENTS.md)
# ---------------------------------------------------------------------------

def market_frame(ticker):
    """one row per trading day: the price FEATURES, plus target5 = the return over the next 5 days.
    target5 is NaN for the latest 5 days because their outcome isn't known yet."""
    # the stock's prices, plus SPY to measure what the whole market did.
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
        # the answer we try to predict: the % change from today's close to the close 5 days later.
        "target5": close.shift(-5) / close - 1,
    })
    # remove broken values and the first days that don't have enough history yet.
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)


def _join_headlines(titles):
    """glue one day's headlines into a single block of text."""
    return " · ".join(titles.astype(str))


def _similar_news_days(news, known, top_k=15):
    """find the past days whose headlines are most similar (TF-IDF cosine) to today's 10 newest
    headlines, and what the stock did over the following 5 days."""
    # label each headline with its calendar day.
    news = news.assign(day=news.date.dt.normalize())
    # "today's story" = the 10 newest headlines.
    latest = news.sort_values("date", ascending=False).head(10)
    # candidate days: outcome already known, and not one of the days the query headlines come from.
    past = news[news.day.isin(known.index) & ~news.day.isin(latest.day)]
    day_docs = past.groupby("day").title.apply(_join_headlines)  # one text per day
    if day_docs.empty:
        return []

    # turn each day's text into word weights (TF-IDF), then measure how close every past day is
    # to today (cosine similarity: 1 = same words, 0 = nothing in common).
    query = " ".join(latest.title.astype(str))
    vectors = TfidfVectorizer(max_features=3000, stop_words="english").fit_transform([*day_docs, query])
    similarity = cosine_similarity(vectors[-1], vectors[:-1]).ravel()
    # keep the 15 closest days and what the stock did over the 5 days after each one.
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
    """dashboard direction forecast: 70% price model + 30% similarity-weighted outcome of the
    most similar past news days."""
    # part 1: a ridge regression on the price features, trained on every day whose outcome we know.
    known = frame.dropna(subset=["target5"])  # days whose 5-day outcome is known
    price_model = ridge_model().fit(known[FEATURES], known.target5)
    base = float(price_model.predict(frame[FEATURES].iloc[[-1]])[0])  # forecast from today's features

    # part 2: the average outcome of the most similar past news days (closer matches count more).
    matches = _similar_news_days(news, known)
    if matches:
        outcomes = [m["actual_5d"] for m in matches]
        weights = [max(m["similarity"], 0.001) for m in matches]
        news_estimate = np.average(outcomes, weights=weights)
    else:
        news_estimate = base

    # blend the two: 70% price model + 30% news. (in testing this did not beat "always up".)
    return {
        "prediction": 0.7 * base + 0.3 * news_estimate,
        "market_baseline": base,
        "momentum_5d": float(frame.mom5.iloc[-1]),
        "matches": matches,
        "positive_match_rate": float(np.mean([m["actual_5d"] > 0 for m in matches])) if matches else None,
        "news_count": len(news),
    }
