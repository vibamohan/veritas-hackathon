"""Score dated headlines with an optional OpenAI-compatible chat model.

Set OPENAI_API_KEY (and optionally OPENAI_MODEL / OPENAI_BASE_URL), then run:
  uv run python scripts/llm_news.py --ticker NVDA

Scores are cached by ticker and title in data/cache/llm_news_scores.parquet. The
walk-forward experiment runner only consumes scores dated before each forecast.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echomarket import CACHE, TICKERS, dataset_news, webhose_news

BATCH_SIZE = 20
OUTPUT = CACHE / "llm_news_scores.parquet"


def score_batch(ticker, titles, api_key, model, base_url):
    indexed = "\n".join(f'{i}: {json.dumps(title)}' for i, title in enumerate(titles))
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Classify each financial news headline's expected near-term impact on the named ticker. "
                        "Return JSON with a scores array in the same order; each item has sentiment (-1 bearish, "
                        "0 neutral/unclear, +1 bullish) and confidence (0 to 1). Judge the headline, not general "
                        "company quality. Do not infer facts beyond the headline."
                    ),
                },
                {"role": "user", "content": f"Ticker: {ticker}\nHeadlines:\n{indexed}"},
            ],
        },
        timeout=90,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    values = json.loads(content)["scores"]
    if len(values) != len(titles):
        raise ValueError("LLM returned a different number of scores than input headlines")
    return [
        {
            "ticker": ticker,
            "title": title,
            "llm_sentiment": max(-1.0, min(1.0, float(value["sentiment"]))),
            "llm_confidence": max(0.0, min(1.0, float(value["confidence"]))),
        }
        for title, value in zip(titles, values)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", choices=TICKERS, action="append", help="Repeat to score multiple tickers; default: all")
    parser.add_argument("--limit", type=int, help="Optional maximum uncached titles per ticker")
    args = parser.parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("OPENAI_API_KEY is required; no headlines were sent")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    cache = pd.read_parquet(OUTPUT) if OUTPUT.exists() else pd.DataFrame(
        columns=["ticker", "title", "llm_sentiment", "llm_confidence"]
    )
    done = set(zip(cache.ticker.astype(str), cache.title.astype(str)))
    additions = []

    for ticker in args.ticker or TICKERS:
        news = pd.concat([dataset_news(ticker), webhose_news(ticker)], ignore_index=True)
        titles = list(dict.fromkeys(news.title.dropna().astype(str)))
        missing = [title for title in titles if (ticker, title) not in done]
        if args.limit is not None:
            missing = missing[: max(0, args.limit)]
        print(f"{ticker}: scoring {len(missing)} of {len(titles)} uncached headlines", flush=True)
        for start in range(0, len(missing), BATCH_SIZE):
            batch = missing[start : start + BATCH_SIZE]
            additions.extend(score_batch(ticker, batch, api_key, model, base_url))
            # Persist each batch so an interrupted run can continue from cache.
            cache = pd.concat([cache, pd.DataFrame(additions)], ignore_index=True).drop_duplicates(
                ["ticker", "title"], keep="last"
            )
            additions.clear()
            cache.to_parquet(OUTPUT, index=False)
            time.sleep(0.2)
    print(f"Saved LLM scores to {OUTPUT}")


if __name__ == "__main__":
    main()
