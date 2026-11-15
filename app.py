"""Flask web server: serves the React dashboard and three JSON endpoints.

  /api/health     list of supported tickers
  /api/dashboard  prices, headlines, turbulence outlook and the experimental direction forecast
  /api/replay     the direction model's validation results (train through 2019, validate 2020–21)

Speed: each ticker's /api/dashboard and /api/replay response is cached in memory for 10 minutes,
and all four tickers are computed in a background thread when the server starts, so switching
tickers in the dashboard doesn't wait for downloads or model training.

Run with:  uv run python app.py   then open http://127.0.0.1:5000
Set ECHOMARKET_WARMUP=0 to skip the startup pre-loading (e.g. in tests).
"""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from sklearn.metrics import mean_absolute_error, mean_squared_error

from echomarket import FEATURES, TICKERS, all_news, forecast_bundle, market_frame, prices, ridge_model, valid_ticker

app = Flask(__name__, static_folder="frontend/dist", static_url_path="")


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

RESPONSE_TTL = 600  # seconds a computed response is reused
_responses = {}  # (endpoint, ticker) -> (computed_at, payload)
_response_locks = {}  # (endpoint, ticker) -> lock; created on first use (custom tickers are allowed)
_locks_guard = threading.Lock()


def cached_response(endpoint, ticker, compute):
    """Return the cached payload if it's recent, otherwise compute and cache it.
    The lock makes a second request for the same ticker wait for the first instead of duplicating work."""
    key = (endpoint, ticker)
    with _locks_guard:
        lock = _response_locks.setdefault(key, threading.Lock())
    with lock:
        hit = _responses.get(key)
        if hit and time.time() - hit[0] < RESPONSE_TTL:
            return hit[1]
        payload = compute(ticker)
        _responses[key] = (time.time(), payload)
        return payload


_outlook = {"computed_at": 0.0, "by_ticker": {}}
_outlook_lock = threading.Lock()


def turbulence(ticker):
    """5-day turbulence outlook from scripts/volatility.py, recomputed at most once an hour
    (it trains on all four tickers at once). Returns None if it can't be computed."""
    try:
        with _outlook_lock:
            if time.time() - _outlook["computed_at"] > 3600:
                from scripts.volatility import outlook  # imported lazily: only needed for the dashboard

                _outlook.update(computed_at=time.time(), by_ticker=outlook())
        return _outlook["by_ticker"].get(ticker)
    except Exception as error:
        app.logger.warning("turbulence outlook unavailable: %s", error)
        return None


def warm_up_ticker(ticker):
    for endpoint, compute in (("dashboard", dashboard_payload), ("replay", replay_payload)):
        try:
            cached_response(endpoint, ticker, compute)
        except Exception as error:
            app.logger.warning("warm-up failed for %s %s: %s", endpoint, ticker, error)


def warm_up():
    """Compute every ticker's responses in the background (in parallel), so the first clicks are fast."""
    prices("SPY")  # shared by every ticker: download once before the parallel work starts
    with ThreadPoolExecutor(max_workers=len(TICKERS)) as pool:
        list(pool.map(warm_up_ticker, TICKERS))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def records(df):
    """DataFrame -> list of dicts for JSON (NaN becomes null)."""
    return df.replace({np.nan: None}).to_dict("records")


def requested_ticker():
    """The requested Yahoo Finance symbol, or None if it has invalid characters."""
    ticker = request.args.get("ticker", "NVDA").upper()
    return ticker if valid_ticker(ticker) else None


# ---------------------------------------------------------------------------
# /api/dashboard
# ---------------------------------------------------------------------------

def dashboard_payload(ticker):
    news = all_news(ticker)
    history = prices(ticker).tail(260).reset_index()  # about one year of trading days
    history["Date"] = history.Date.dt.strftime("%Y-%m-%d")
    latest_news = news.sort_values("date", ascending=False).head(40)
    return {
        "ticker": ticker,
        "price_history": records(history[["Date", "Close", "Volume"]]),
        "news": records(latest_news.assign(date=latest_news.date.astype(str))),
        "turbulence": turbulence(ticker),
        **forecast_bundle(market_frame(ticker), news),
    }


# ---------------------------------------------------------------------------
# /api/replay: validation of the direction model
# ---------------------------------------------------------------------------

HORIZON = 5  # trading days; labels look 5 days ahead, so 5 rows are purged at each split boundary


class NotEnoughHistory(Exception):
    pass


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


def replay_payload(ticker):
    frame = market_frame(ticker).dropna()
    X, y = frame[FEATURES], frame.target5
    rows = split_indices(pd.DatetimeIndex(frame.index))
    if len(rows["train"]) < 100 or len(rows["validation"]) < 3:
        raise NotEnoughHistory("Insufficient history for train through 2019 and validate on 2020–2021.")

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
    return {
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
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return send_from_directory("frontend/dist", "index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "tickers": TICKERS, "custom_tickers": True})


@app.get("/api/dashboard")
def dashboard():
    ticker = requested_ticker()
    if not ticker:
        return jsonify({"error": "Unsupported ticker"}), 400
    try:
        return jsonify(cached_response("dashboard", ticker, dashboard_payload))
    except Exception as error:
        return jsonify({"error": str(error)}), 502


@app.get("/api/replay")
def replay():
    ticker = requested_ticker()
    if not ticker:
        return jsonify({"error": "Unsupported ticker"}), 400
    try:
        return jsonify(cached_response("replay", ticker, replay_payload))
    except NotEnoughHistory as error:
        return jsonify({"error": str(error)}), 422
    except Exception as error:
        return jsonify({"error": str(error)}), 502


# Start pre-loading as soon as the module is imported (by `python app.py` or by gunicorn).
if os.environ.get("ECHOMARKET_WARMUP", "1") != "0":
    threading.Thread(target=warm_up, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
