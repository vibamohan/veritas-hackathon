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

The app downloads prices through yfinance and headlines from GDELT, caching files under `data/cache/`. Current prototype uses TF-IDF text similarity and Ridge regression with chronological replay.

Download a few upstream Webhose releases into `data/webhose/` with:

```bash
uv run python scripts/download_webhose_dataset.py --latest 4
```

The downloader selects the newest weekly ZIP archives (positive and negative sentiment combined by default). The app reads JSON, JSONL, and CSV files inside those ZIPs as well as loose CSV/JSON/JSONL files. It extracts publication date, title, text, URL, source, and the archive's sentiment label, then finds ticker mentions for historical matching. Keep the number of archives modest for a quick demo; the repository adds new releases over time. Review the [Webhose dataset repository](https://github.com/Webhose/financial-news-dataset) and its [terms of use](https://github.com/Webhose/free-news-datasets/blob/master/tou.MD) before use: the dataset is described as available for academic, research, and journalistic purposes.

This is an educational research prototype, not investment advice. GDELT coverage and Webhose article ticker/date mapping may be incomplete; predictions are experimental.
