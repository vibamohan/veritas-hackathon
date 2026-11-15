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

Put Webhose dataset CSV, JSON, or JSONL files in `data/webhose/`. Files are parsed for publication date and article text; articles mentioning the selected ticker are included. Review the [Webhose dataset repository](https://github.com/Webhose/financial-news-dataset) terms before use; its README describes its datasets as available for academic, research, and journalistic purposes.

This is an educational research prototype, not investment advice. GDELT coverage and Webhose article ticker/date mapping may be incomplete; predictions are experimental.
