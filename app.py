from flask import Flask, jsonify, request, send_from_directory
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from echomarket import FEATURES, RIDGE_ALPHAS, TICKERS, all_news, forecast_bundle, market_frame, prices, ridge_model

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
        # Fixed date split requested for a simple, auditable validation comparison.
        # Purge five rows at both boundaries because labels span the next five sessions.
        dates = pd.DatetimeIndex(f.index)
        train_mask = dates < "2020-01-01"
        validation_mask = (dates >= "2020-01-01") & (dates < "2022-01-01")
        test_mask = dates >= "2022-01-01"
        train_end = np.flatnonzero(train_mask)[-1] - 5
        val_start = np.flatnonzero(validation_mask)[0] + 5
        val_end = np.flatnonzero(validation_mask)[-1] - 5
        test_start = np.flatnonzero(test_mask)[0] + 5
        train_indices = np.arange(0, train_end + 1)
        validation_indices = np.arange(val_start, val_end + 1, 5)
        test_indices = np.arange(test_start, len(f) - 5, 5)
        if len(train_indices) < 100 or len(validation_indices) < 3:
            return jsonify({"error": "Insufficient history for train through 2019 and validate on 2020–2021."}), 422

        model = ridge_model(1.0).fit(X.iloc[train_indices], y.iloc[train_indices])
        validation_out = []
        for i in validation_indices:
            pred = float(model.predict(X.iloc[[i]])[0])
            validation_out.append({"date": f.index[i].strftime("%Y-%m-%d"), "predicted": pred,
                                   "always_up": 0.01, "actual": float(y.iloc[i])})

        # Keep the specified 2022+ test period separate and unscored in this comparison.
        test_out = []
        if len(test_indices):
            final_model = ridge_model(1.0).fit(X.iloc[:np.flatnonzero(test_mask)[0] - 5], y.iloc[:np.flatnonzero(test_mask)[0] - 5])
            for i in test_indices:
                test_out.append({"date": f.index[i].strftime("%Y-%m-%d"),
                                 "predicted": float(final_model.predict(X.iloc[[i]])[0]),
                                 "actual": float(y.iloc[i])})

        def metrics(predictions):
            actuals = [x["actual"] for x in validation_out]
            if not validation_out:
                return None
            directional = [np.sign(p) == np.sign(a) for p, a in zip(predictions, actuals) if p != 0]
            return {
                "mae": float(mean_absolute_error(actuals, predictions)),
                "rmse": float(mean_squared_error(actuals, predictions) ** 0.5),
                "directional_accuracy": float(np.mean(directional)) if directional else None,
                "mean_prediction": float(np.mean(predictions)),
                "mean_actual": float(np.mean(actuals)),
                "prediction_actual_correlation": float(np.corrcoef(predictions, actuals)[0, 1]) if np.std(predictions) > 0 and np.std(actuals) > 0 else None,
                "samples": len(validation_out),
            }

        return jsonify({
            "ticker": ticker,
            "split": {"method": "fixed chronological dates; five-row purging", "train_through": f.index[train_end].strftime("%Y-%m-%d"),
                      "validation_start": validation_out[0]["date"], "validation_end": validation_out[-1]["date"],
                      "test_start": test_out[0]["date"] if test_out else None,
                      "horizon_trading_days": 5, "validation_stride_trading_days": 5},
            "architecture": "StandardScaler -> Ridge(alpha=1) on six lagged market features",
            "tuning": {"method": "fixed current model; requested validation comparison", "selected_alpha": 1.0},
            "metrics": {"current_model": metrics([x["predicted"] for x in validation_out]),
                        "always_up_baseline": metrics([x["always_up"] for x in validation_out])},
            "forecasts": validation_out,
            "test_forecasts": test_out,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 502


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
