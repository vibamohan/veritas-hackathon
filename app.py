"""Flask web server: serves the React dashboard and three JSON endpoints.

  /api/health     list of supported tickers
  /api/dashboard  prices, headlines, turbulence outlook and the experimental direction forecast
  /api/replay     the direction model's validation results (train through 2019, validate 2020–21)

Run with:  uv run python app.py   then open http://127.0.0.1:5000
"""
import time

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from sklearn.metrics import mean_absolute_error, mean_squared_error

from echomarket import FEATURES, TICKERS, all_news, forecast_bundle, market_frame, prices, ridge_model

app = Flask(__name__, static_folder="frontend/dist", static_url_path="")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_outlook_cache = {"computed_at": 0.0, "by_ticker": {}}


def turbulence(ticker):
    """5-day turbulence outlook from scripts/volatility.py, recomputed at most once an hour
    (it trains on all four tickers). Returns None if it can't be computed."""
    try:
        if time.time() - _outlook_cache["computed_at"] > 3600:
            from scripts.volatility import outlook  # imported lazily: only needed for the dashboard

            _outlook_cache.update(computed_at=time.time(), by_ticker=outlook())
        return _outlook_cache["by_ticker"].get(ticker)
    except Exception as error:
        app.logger.warning("turbulence outlook unavailable: %s", error)
        return None


def records(df):
    """DataFrame -> list of dicts for JSON (NaN becomes null)."""
    return df.replace({np.nan: None}).to_dict("records")


def requested_ticker():
    """The ?ticker= query parameter, or None if it isn't one we support."""
    ticker = request.args.get("ticker", "NVDA").upper()
    return ticker if ticker in TICKERS else None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return send_from_directory("frontend/dist", "index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "tickers": TICKERS})


@app.get("/api/dashboard")
def dashboard():
    ticker = requested_ticker()
    if not ticker:
        return jsonify({"error": "Unsupported ticker"}), 400
    try:
        news = all_news(ticker)
        history = prices(ticker).tail(260).reset_index()  # about one year of trading days
        history["Date"] = history.Date.dt.strftime("%Y-%m-%d")
        latest_news = news.sort_values("date", ascending=False).head(40)
        return jsonify({
            "ticker": ticker,
            "price_history": records(history[["Date", "Close", "Volume"]]),
            "news": records(latest_news.assign(date=latest_news.date.astype(str))),
            "turbulence": turbulence(ticker),
            **forecast_bundle(market_frame(ticker), news),
        })
    except Exception as error:
        return jsonify({"error": str(error)}), 502


# --- /api/replay: validation of the direction model ---------------------------

HORIZON = 5  # trading days; labels look 5 days ahead, so 5 rows are purged at each split boundary


def split_indices(dates):
    """Row numbers for a fixed chronological split: train through 2019, validate 2020–21, test 2022+.
    Validation and test rows are spaced 5 days apart so scored outcomes don't overlap."""
    train = np.flatnonzero(dates < "2020-01-01")
    validation = np.flatnonzero((dates >= "2020-01-01") & (dates < "2022-01-01"))
    test = np.flatnonzero(dates >= "2022-01-01")
    return {
        "train": np.arange(0, train[-1] - HORIZON + 1),
        "validation": np.arange(validation[0] + HORIZON, validation[-1] - HORIZON + 1, HORIZON),
        "test": np.arange(test[0] + HORIZON, len(dates) - HORIZON, HORIZON),
        "test_train_end": test[0] - HORIZON,  # rows before this are available when training for the test
    }


def score(predictions, actuals):
    """Error and direction metrics for one set of forecasts."""
    if not actuals:
        return None
    hits = [np.sign(p) == np.sign(a) for p, a in zip(predictions, actuals) if p != 0]
    correlated = np.std(predictions) > 0 and np.std(actuals) > 0
    return {
        "mae": float(mean_absolute_error(actuals, predictions)),
        "rmse": float(mean_squared_error(actuals, predictions) ** 0.5),
        "directional_accuracy": float(np.mean(hits)) if hits else None,
        "mean_prediction": float(np.mean(predictions)),
        "mean_actual": float(np.mean(actuals)),
        "prediction_actual_correlation": float(np.corrcoef(predictions, actuals)[0, 1]) if correlated else None,
        "samples": len(actuals),
    }


@app.get("/api/replay")
def replay():
    ticker = requested_ticker()
    if not ticker:
        return jsonify({"error": "Unsupported ticker"}), 400
    try:
        frame = market_frame(ticker).dropna()
        X, y = frame[FEATURES], frame.target5
        rows = split_indices(pd.DatetimeIndex(frame.index))
        if len(rows["train"]) < 100 or len(rows["validation"]) < 3:
            return jsonify({"error": "Insufficient history for train through 2019 and validate on 2020–2021."}), 422

        def day(i):
            return frame.index[i].strftime("%Y-%m-%d")

        # Validation: train once on pre-2020 data, forecast every 5th day of 2020–21.
        model = ridge_model(1.0).fit(X.iloc[rows["train"]], y.iloc[rows["train"]])
        validation = [
            {"date": day(i), "predicted": float(model.predict(X.iloc[[i]])[0]), "always_up": 0.01, "actual": float(y.iloc[i])}
            for i in rows["validation"]
        ]

        # Test (2022+): returned for reference but deliberately not scored, to avoid tuning on it.
        end = rows["test_train_end"]
        test_model = ridge_model(1.0).fit(X.iloc[:end], y.iloc[:end])
        test = [
            {"date": day(i), "predicted": float(test_model.predict(X.iloc[[i]])[0]), "actual": float(y.iloc[i])}
            for i in rows["test"]
        ]

        actuals = [v["actual"] for v in validation]
        return jsonify({
            "ticker": ticker,
            "split": {
                "method": "fixed chronological dates; five-row purging",
                "train_through": day(rows["train"][-1]),
                "validation_start": validation[0]["date"],
                "validation_end": validation[-1]["date"],
                "test_start": test[0]["date"] if test else None,
                "horizon_trading_days": HORIZON,
                "validation_stride_trading_days": HORIZON,
            },
            "architecture": "StandardScaler -> Ridge(alpha=1) on six lagged market features",
            "tuning": {"method": "fixed current model; requested validation comparison", "selected_alpha": 1.0},
            "metrics": {
                "current_model": score([v["predicted"] for v in validation], actuals),
                "always_up_baseline": score([v["always_up"] for v in validation], actuals),
            },
            "forecasts": validation,
            "test_forecasts": test,
        })
    except Exception as error:
        return jsonify({"error": str(error)}), 502


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
