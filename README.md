# EchoMarket

Live-first historical event similarity prototype for NVDA, TSLA, AAPL, and SPY.

## Run

```bash
uv sync
npm --prefix frontend install
npm --prefix frontend run build
uv run python app.py
```

Open http://127.0.0.1:5000. Flask serves the React production build and exposes `/api/dashboard`, `/api/replay`, and `/api/health`. For frontend development, run `npm --prefix frontend run dev` in another terminal; Vite proxies API calls to Flask.

The app downloads prices through yfinance (full history) and live headlines from GDELT, caching files under `data/cache/`. Current prototype uses TF-IDF text similarity and a StandardScaler + Ridge regression on six lagged market features. Data loading and inference live in `echomarket.py`; `app.py` is the Flask API. The replay endpoint uses fixed chronological dates (train through 2019, validate 2020–2021, test 2022 onward), with a five-trading-day purge at split boundaries; it compares the current model with an always-up baseline on validation using MAE, RMSE, and directional accuracy. See [docs/MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md) for architecture and limitations.

## Historical news datasets

Ready-to-use headline files for our tickers are committed in `data/datasets/` (date, title, url, source, ticker), so a fresh clone works with no downloads. A row is kept if the dataset tags it with the ticker or its headline names the company (e.g. "Nvidia", "S&P 500").

| File | Source | Coverage | License |
| --- | --- | --- | --- |
| `huggingface.csv` | [oliverwang15/us_stock_news_with_price](https://huggingface.co/datasets/oliverwang15/us_stock_news_with_price) | 2009–2023 | not stated |
| `kaggle.csv` | [Massive Stock News Analysis DB](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests) (Miguel Aenlle) | 2009–2020 | CC0 |
| `fliarbi.csv` | [Fliar BI Corporate Events 2019–2026](https://fliarbi.com/datasets/corporate-events-2019-2026/), free 7% sample | 2019–2026 | CC BY-NC 4.0 |

To rebuild them, run `uv run python scripts/download_datasets.py`. Kaggle needs a login: download its ZIP manually and save it as `data/datasets/kaggle.zip` first (it is git-ignored), otherwise the committed `kaggle.csv` is kept.

## Webhose (optional)

Download a few upstream Webhose releases into `data/webhose/` with:

```bash
uv run python scripts/download_webhose_dataset.py --latest 4
```

The downloader selects the newest weekly ZIP archives (positive and negative sentiment combined by default). The app reads JSON, JSONL, and CSV files inside those ZIPs as well as loose CSV/JSON/JSONL files. It extracts publication date, title, URL, and source, keeping articles that mention the ticker or company name. Keep the number of archives modest for a quick demo; the repository adds new releases over time. Review the [Webhose dataset repository](https://github.com/Webhose/financial-news-dataset) and its [terms of use](https://github.com/Webhose/free-news-datasets/blob/master/tou.MD) before use: the dataset is described as available for academic, research, and journalistic purposes.

This is an educational research prototype, not investment advice. GDELT coverage and Webhose article ticker/date mapping may be incomplete; predictions are experimental.

## Model experiments

Run `uv run python scripts/experiments.py --target excess` to compare regression,
classification, and point-in-time news features with walk-forward validation. The
runner includes Ridge, ElasticNet, random forest, extra trees, gradient boosting,
TF-IDF neighbors, and optional sentence embeddings. See
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the evaluation design and prior results.

Optional LLM headline scoring is a separate, cached step. Set `OPENAI_API_KEY` and
run `uv run python scripts/llm_news.py --ticker NVDA` (repeat `--ticker` as needed),
then rerun `scripts/experiments.py`. `OPENAI_MODEL` and `OPENAI_BASE_URL` can select
another compatible endpoint. Scoring makes external model requests; without cached
scores the LLM feature models are skipped. Never include future outcomes in the
headline prompt or score cache.
