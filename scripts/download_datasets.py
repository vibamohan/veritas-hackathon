"""Rebuild data/datasets/*.csv (date, title, url, source, ticker) from the raw datasets.

The CSVs are committed, so collaborators only need this to refresh or change the data.
Hugging Face and Fliar BI download automatically; Kaggle needs a login, so save its ZIP
as data/datasets/kaggle.zip first (otherwise the existing kaggle.csv is kept).

Usage: uv run python scripts/download_datasets.py
"""
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echomarket import TICKERS, mentions

DEST = Path("data/datasets")
HF = "https://huggingface.co/datasets/oliverwang15/us_stock_news_with_price/resolve/main/data/{}-00000-of-00001.parquet"
FLIARBI = "https://huggingface.co/datasets/fliarbi/corporate-events-2019-2026/resolve/main/data/train.parquet"
KAGGLE_PAGE = "https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests"


def get(url):
    """Download a file into memory."""
    print("downloading", url)
    response = requests.get(url, timeout=600)
    response.raise_for_status()
    return io.BytesIO(response.content)


def save(df, name):
    """Keep only rows about our tickers and write data/datasets/<name>.csv.
    A row belongs to a ticker if the dataset tags it with that ticker OR its headline names it
    (so a headline can be kept for several tickers)."""
    per_ticker = []
    for t in TICKERS:
        about_t = (df.ticker == t) | df.title.str.contains(mentions(t), na=False)
        per_ticker.append(df[about_t].assign(ticker=t))
    df = pd.concat(per_ticker)
    df["date"] = df.date.astype(str).str[:10]  # keep the calendar day, drop time/timezone
    df = df.drop_duplicates(["ticker", "title"])
    df.to_csv(DEST / f"{name}.csv", index=False)
    print(f"saved {len(df)} rows to {DEST / name}.csv:", df.ticker.value_counts().to_dict())


# Each loader renames the source's columns to: date, ticker, title, url, source.

def huggingface():
    cols = {"date": "date", "stock": "ticker", "title": "title"}
    df = pd.concat(pd.read_parquet(get(HF.format(s)), columns=list(cols)) for s in ("train", "test"))
    save(df.rename(columns=cols).assign(url="", source="huggingface"), "huggingface")


def kaggle():
    # Kaggle needs a login: download the ZIP from KAGGLE_PAGE and save it as data/datasets/kaggle.zip
    path = DEST / "kaggle.zip"
    if not path.exists():
        print(f"skipping Kaggle: download the ZIP from {KAGGLE_PAGE} and save it as {path}")
        return
    cols = {"date": "date", "stock": "ticker", "headline": "title", "url": "url", "publisher": "source"}
    with zipfile.ZipFile(path) as z:
        df = pd.concat(
            pd.read_csv(z.open(f), usecols=list(cols)) for f in ("raw_analyst_ratings.csv", "raw_partner_headlines.csv")
        )
    save(df.rename(columns=cols), "kaggle")


def fliarbi():
    # Free 7% sample (CC BY-NC 4.0). Rows have no ticker, so only headline mentions count.
    d = pd.read_parquet(get(FLIARBI), columns=["event_date", "headline", "news_url", "news_source"])
    d.columns = ["date", "title", "url", "source"]
    save(d.assign(ticker=""), "fliarbi")


if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    huggingface()
    kaggle()
    fliarbi()
