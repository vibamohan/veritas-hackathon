"""Flask web server: serves the React dashboard and three JSON endpoints.

  /api/health     list of supported tickers
  /api/dashboard  prices, headlines, turbulence outlook and the experimental direction forecast
  /api/replay     the direction model's validation results (train through 2019, validate 2020–21)

Speed and reliability: /api/dashboard and /api/replay responses are cached in memory. A background
thread fills the cache for the four preset tickers shortly after startup and refreshes it every
10 minutes. Computations run in a small background pool, and a request never waits more than
45 seconds: it gets cached data or a "still loading" (503) reply that the page retries.

Run with:  uv run python app.py   then open http://127.0.0.1:5000
Set ECHOMARKET_WARMUP=0 to skip the startup pre-loading (e.g. in tests), and
ECHOMARKET_WARMUP_DELAY to change how long it waits after startup (default 20 seconds).
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
# Caching and background computation
#
# Heavy work (downloads, model training) never runs on the threads that answer requests. It runs
# in a small pool of COMPUTE_WORKERS threads, and a request waits at most REQUEST_WAIT seconds for it.
# If a download is slow or stuck (e.g. Yahoo rate-limiting the server), visitors get the last known
# data or a "still loading" reply instead of a frozen site, and /api/health always answers.
# ---------------------------------------------------------------------------

RESPONSE_TTL = 1800  # seconds a computed response counts as fresh
REFRESH_EVERY = 600  # seconds between background refreshes of the 4 preset tickers
REQUEST_WAIT = 45  # max seconds a request waits for a computation before replying "still loading"
COMPUTE_WORKERS = 2  # parallel computations (Render's free tier has 0.1 CPU and 512 MB)

_compute_pool = ThreadPoolExecutor(max_workers=COMPUTE_WORKERS, thread_name_prefix="compute")
_responses = {}  # (endpoint, ticker) -> (computed_at, payload)
_running = {}  # (endpoint, ticker) -> Future of the computation in progress
_running_lock = threading.Lock()


def log(message):
    print(f"[echomarket] {message}", flush=True)  # shows up in the Render log


def _compute_and_store(key, compute):
    endpoint, ticker = key
    started = time.time()
    try:
        payload = compute(ticker)
    except Exception as error:
        log(f"{endpoint} {ticker} failed after {time.time() - started:.1f}s: {error}")
        raise
    _responses[key] = (time.time(), payload)
    log(f"{endpoint} {ticker} computed in {time.time() - started:.1f}s")
    return payload


def start_compute(endpoint, ticker, compute):
    """Start (re)computing a response in the background pool, unless it's already running."""
    key = (endpoint, ticker)
    with _running_lock:
        future = _running.get(key)
        if future is None or future.done():
            future = _compute_pool.submit(_compute_and_store, key, compute)
            _running[key] = future
        return future


def cached_response(endpoint, ticker, compute):
    """Fresh cached payload if there is one. Otherwise start a recomputation and either return the
    older payload right away, or wait up to REQUEST_WAIT seconds (raises TimeoutError after that)."""
    hit = _responses.get((endpoint, ticker))
    if hit and time.time() - hit[0] < RESPONSE_TTL:
        return hit[1]
    future = start_compute(endpoint, ticker, compute)
    if hit:
        return hit[1]  # slightly old data now; the next visitor gets the refreshed version
    return future.result(timeout=REQUEST_WAIT)


_outlook = {"computed_at": 0.0, "by_ticker": {}}
_outlook_lock = threading.Lock()


def turbulence(ticker):
    """5-day turbulence outlook from scripts/volatility.py, recomputed at most once an hour
    (it trains on all four tickers at once). Returns None if it can't be computed."""
    if not _outlook_lock.acquire(timeout=REQUEST_WAIT):  # another thread is (slowly) computing it
        return _outlook["by_ticker"].get(ticker)
    try:
        if time.time() - _outlook["computed_at"] > 3600:
            from scripts.volatility import outlook  # imported lazily: only needed for the dashboard

            started = time.time()
            _outlook.update(computed_at=time.time(), by_ticker=outlook())
            log(f"turbulence outlook computed in {time.time() - started:.1f}s")
        return _outlook["by_ticker"].get(ticker)
    except Exception as error:
        log(f"turbulence outlook unavailable: {error}")
        return None
    finally:
        _outlook_lock.release()


WARMUP_DELAY = int(os.environ.get("ECHOMARKET_WARMUP_DELAY", "20"))  # seconds


def keep_warm():
    """Background loop: fill the cache for the 4 preset tickers, then refresh it every 10 minutes.
    Gentle on purpose for small hosts: it waits until the server is answering requests and health
    checks, then refreshes one response at a time. Visitors keep getting the previous responses
    while a refresh runs."""
    time.sleep(WARMUP_DELAY)
    while True:
        for ticker in TICKERS:
            for endpoint, compute in (("dashboard", dashboard_payload), ("replay", replay_payload)):
                try:
                    start_compute(endpoint, ticker, compute).result(timeout=300)
                except Exception as error:  # includes a timeout: move on, try again next round
                    log(f"background refresh of {endpoint} {ticker} did not finish: {error!r}")
        time.sleep(REFRESH_EVERY)


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


def still_loading(ticker):
    """503 reply telling the page to retry shortly (the computation keeps running meanwhile)."""
    return jsonify({"error": f"Still loading {ticker}: the server is busy downloading data. Retrying…",
                    "loading": True}), 503


@app.get("/api/dashboard")
def dashboard():
    ticker = requested_ticker()
    if not ticker:
        return jsonify({"error": "Unsupported ticker"}), 400
    try:
        return jsonify(cached_response("dashboard", ticker, dashboard_payload))
    except TimeoutError:
        return still_loading(ticker)
    except Exception as error:
        return jsonify({"error": str(error)}), 502


@app.get("/api/replay")
def replay():
    ticker = requested_ticker()
    if not ticker:
        return jsonify({"error": "Unsupported ticker"}), 400
    try:
        return jsonify(cached_response("replay", ticker, replay_payload))
    except TimeoutError:
        return still_loading(ticker)
    except NotEnoughHistory as error:
        return jsonify({"error": str(error)}), 422
    except Exception as error:
        return jsonify({"error": str(error)}), 502


# Start pre-loading as soon as the module is imported (by `python app.py` or by gunicorn).
if os.environ.get("ECHOMARKET_WARMUP", "1") != "0":
    threading.Thread(target=keep_warm, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
