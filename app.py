from flask import Flask, jsonify, request, send_from_directory
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from echomarket import FEATURES, TICKERS, all_news, forecast_bundle, market_frame, prices

app = Flask(__name__, static_folder="frontend/dist", static_url_path="")


def records(df):
    return df.replace({np.nan: None}).to_dict("records")


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
        news = all_news(ticker)
        hist = prices(ticker).tail(260).reset_index()
        hist["Date"] = hist.Date.dt.strftime("%Y-%m-%d")
        latest = news.sort_values("date", ascending=False).head(40)
        return jsonify(
            {
                "ticker": ticker,
                "price_history": records(hist[["Date", "Close", "Volume"]]),
                "news": records(latest.assign(date=latest.date.astype(str))),
                **forecast_bundle(market_frame(ticker), news),
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
        f = market_frame(ticker).dropna()
        X, y = f[FEATURES], f.target5
        out = []
        for i in range(max(180, len(f) // 2), len(f), 10):
            # Train only on days whose 5-day outcome was already known on day i.
            pred = float(Ridge(alpha=1).fit(X.iloc[: i - 5], y.iloc[: i - 5]).predict(X.iloc[[i]])[0])
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
                "mae": mean_absolute_error([x["actual"] for x in out], [x["predicted"] for x in out]) if out else None,
                "directional_accuracy": float(np.mean([x["correct_direction"] for x in out])) if out else None,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 502


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
