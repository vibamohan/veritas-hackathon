from __future__ import annotations
from flask import Flask, jsonify, request, send_from_directory
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from echomarket import (
    TICKERS,
    prices,
    gdelt,
    webhose_news,
    market_frame,
    forecast_bundle,
)

app = Flask(__name__, static_folder="frontend/dist", static_url_path="")


@app.get("/")
def index():
    return send_from_directory("frontend/dist", "index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "tickers": TICKERS})


@app.get("/api/dashboard")
def dashboard():
    ticker = request.args.get("ticker", "NVDA").upper()
    if ticker not in TICKERS:
        return jsonify({"error": "Unsupported ticker"}), 400
    try:
        raw = prices(ticker)
        frame = market_frame(ticker)
        news = gdelt(ticker)
        archive = webhose_news(ticker)
        if not archive.empty:
            news = pd.concat([news, archive], ignore_index=True).drop_duplicates(
                "title"
            )
        payload = forecast_bundle(ticker, frame, news)
        hist = raw.reset_index()
        hist["Date"] = pd.to_datetime(hist.Date).dt.strftime("%Y-%m-%d")
        return jsonify(
            {
                "ticker": ticker,
                "price_history": hist.tail(260)[["Date", "Close", "Volume"]]
                .replace({np.nan: None})
                .to_dict("records"),
                "news": news.sort_values("date", ascending=False)
                .head(40)
                .assign(date=lambda x: x.date.astype(str))
                .replace({np.nan: None})
                .to_dict("records"),
                **payload,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.get("/api/replay")
def replay():
    ticker = request.args.get("ticker", "NVDA").upper()
    if ticker not in TICKERS:
        return jsonify({"error": "Unsupported ticker"}), 400
    try:
        f = market_frame(ticker)
        cols = ["ret1", "mom5", "mom20", "vol20", "vol_ratio", "spy5"]
        X = f[cols]
        y = f.target5
        start = max(180, len(f) // 2)
        out = []
        for i in range(start, len(f) - 5, 10):
            m = Ridge(alpha=1).fit(X.iloc[:i], y.iloc[:i])
            pred = float(m.predict(X.iloc[[i]])[0])
            actual = float(y.iloc[i])
            out.append(
                {
                    "date": f.index[i].strftime("%Y-%m-%d"),
                    "predicted": pred,
                    "actual": actual,
                    "correct_direction": bool(np.sign(pred) == np.sign(actual)),
                }
            )
        return jsonify(
            {
                "ticker": ticker,
                "forecasts": out,
                "mae": float(
                    mean_absolute_error(
                        [x["actual"] for x in out], [x["predicted"] for x in out]
                    )
                )
                if out
                else None,
                "directional_accuracy": float(
                    np.mean([x["correct_direction"] for x in out])
                )
                if out
                else None,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 502


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
