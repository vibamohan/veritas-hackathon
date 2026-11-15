// EchoMarket dashboard (React).
//
// App loads two JSON endpoints from the Flask server (app.py) whenever the ticker changes:
//   /api/dashboard  prices, headlines, turbulence outlook, experimental direction forecast
//   /api/replay     validation results for the direction model
// and renders one component per panel. Charts are drawn with Plotly into <div id="..."> elements.

import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import Plotly from "plotly.js-dist-min";
import { Activity, RefreshCw, TrendingUp, Newspaper, History, ChartNoAxesCombined, Waves } from "lucide-react";
import "./style.css";

const TICKERS = ["NVDA", "TSLA", "AAPL", "SPY"];
const PLOT_CONFIG = { displayModeBar: false, responsive: true };

// ---------------------------------------------------------------------------
// Formatting helpers (values from the API are fractions: 0.05 means 5%)
// ---------------------------------------------------------------------------

function pct(x) {
  return x == null ? "—" : `${(x * 100).toFixed(2)}%`;
}
function money(x) {
  return x == null ? "—" : `$${Number(x).toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
}
function signedPct(x) {
  return `${x >= 0 ? "+" : ""}${(x * 100).toFixed(0)}%`;
}
function wholePct(x) {
  return `${(x * 100).toFixed(0)}%`;
}
function plusMinusPct(x) {
  return `±${(x * 100).toFixed(1)}%`;
}
function plusMinusDollars(x) {
  return `±$${Math.round(x).toLocaleString()}`;
}
function plural(n, word) {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

// ---------------------------------------------------------------------------
// Charts (Plotly)
// ---------------------------------------------------------------------------

function drawPriceChart(history) {
  const line = {
    x: history.map((row) => row.Date),
    y: history.map((row) => row.Close),
    type: "scatter",
    mode: "lines",
    line: { color: "#3478f6", width: 2 },
    fill: "tozeroy",
    fillcolor: "rgba(52,120,246,.08)",
  };
  const layout = {
    margin: { l: 45, r: 15, t: 10, b: 35 },
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    xaxis: { gridcolor: "#eef1f5" },
    yaxis: { gridcolor: "#eef1f5", tickprefix: "$" },
  };
  Plotly.newPlot("priceChart", [line], layout, PLOT_CONFIG);
}

// Half-circle gauge for the direction forecast, from -10% to +10%.
function drawGauge(prediction) {
  const value = prediction || 0;
  const gauge = {
    type: "indicator",
    mode: "gauge+number",
    value: value * 100,
    number: { suffix: "%", font: { size: 27, color: "#172b4d" } },
    gauge: {
      axis: { range: [-10, 10], ticksuffix: "%" },
      bar: { color: value >= 0 ? "#19aa83" : "#e05b68" },
      bgcolor: "#f4f7fb",
      borderwidth: 0,
      steps: [
        { range: [-10, 0], color: "#fff0f1" },
        { range: [0, 10], color: "#e8faf4" },
      ],
    },
  };
  Plotly.newPlot("gauge", [gauge], { margin: { t: 18, b: 5, l: 20, r: 20 }, paper_bgcolor: "white" }, PLOT_CONFIG);
}

// Validation forecasts vs what actually happened, plus the "always up" baseline.
function drawReplayChart(forecasts) {
  const dates = forecasts.map((f) => f.date);
  const series = [
    { name: "Actual", key: "actual", mode: "lines+markers", line: { color: "#21a582" } },
    { name: "Current model", key: "predicted", mode: "lines", line: { color: "#3478f6" } },
    { name: "Always up (+1%)", key: "always_up", mode: "lines", line: { color: "#f2a541", dash: "dot" } },
  ].map((s) => ({
    x: dates,
    y: forecasts.map((f) => f[s.key] * 100),
    name: s.name,
    type: "scatter",
    mode: s.mode,
    line: s.line,
  }));
  const layout = {
    margin: { l: 45, r: 10, t: 5, b: 35 },
    legend: { orientation: "h" },
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    xaxis: { gridcolor: "#eef1f5" },
    yaxis: { gridcolor: "#eef1f5", ticksuffix: "%" },
  };
  Plotly.newPlot("replayChart", series, layout, PLOT_CONFIG);
}

// ---------------------------------------------------------------------------
// App: state, data loading, layout
// ---------------------------------------------------------------------------

function App() {
  const [ticker, setTicker] = useState("NVDA");
  const [data, setData] = useState(null); // /api/dashboard response
  const [replay, setReplay] = useState(null); // /api/replay response
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function fetchJson(url) {
    const response = await fetch(url);
    const body = await response.json();
    if (!response.ok) throw Error(body.error);
    return body;
  }

  async function load() {
    setLoading(true);
    setError("");
    try {
      setData(await fetchJson(`/api/dashboard?ticker=${ticker}`));
      setReplay(await fetchJson(`/api/replay?ticker=${ticker}`));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  // Reload whenever the ticker changes.
  useEffect(() => {
    load();
  }, [ticker]);

  // Redraw the charts whenever new data arrives.
  useEffect(() => {
    if (!data) return;
    drawPriceChart(data.price_history || []);
    drawGauge(data.prediction);
    if (replay?.forecasts) drawReplayChart(replay.forecasts);
  }, [data, replay]);

  return (
    <>
      <Sidebar ticker={ticker} onChange={setTicker} />
      <main>
        <header>
          <div>
            <div className="eyebrow">MARKET OVERVIEW / {new Date().toLocaleDateString()}</div>
            <h1>{ticker} Intelligence</h1>
          </div>
          <button onClick={load}>
            <RefreshCw size={15} />
            {loading ? "Loading…" : "Refresh data"}
          </button>
        </header>
        {error && <div className="error">{error}</div>}
        <MetricTiles data={data} />
        <Turbulence t={data?.turbulence} ticker={ticker} />
        <section className="grid">
          <PricePanel />
          <ForecastPanel data={data} />
          <MatchesPanel matches={data?.matches || []} />
          <ReplayPanel replay={replay} />
        </section>
        <HeadlinesPanel news={data?.news || []} count={data?.news_count} />
        <footer>Experimental educational prototype · Data may be delayed, incomplete, or inaccurate.</footer>
      </main>
    </>
  );
}

// ---------------------------------------------------------------------------
// Layout pieces
// ---------------------------------------------------------------------------

function Sidebar({ ticker, onChange }) {
  const [customTicker, setCustomTicker] = useState("");
  const [tickerError, setTickerError] = useState("");

  function submitCustomTicker(event) {
    event.preventDefault();
    const symbol = customTicker.trim().toUpperCase();
    if (!/^[A-Z0-9][A-Z0-9.-]{0,14}$/.test(symbol)) {
      setTickerError("Enter a valid Yahoo Finance symbol.");
      return;
    }
    setTickerError("");
    onChange(symbol);
  }

  return (
    <aside>
      <div className="brand">
        ◉ Echo<span>Market</span>
      </div>
      <p className="muted">HISTORICAL EVENT INTELLIGENCE</p>
      <label>ASSET</label>
      <select value={ticker} onChange={(e) => onChange(e.target.value)}>
        {TICKERS.map((t) => (
          <option key={t}>{t}</option>
        ))}
        {!TICKERS.includes(ticker) && <option value={ticker}>{ticker}</option>}
      </select>
      <form className="custom-ticker" onSubmit={submitCustomTicker}>
        <label htmlFor="customTicker">OR ENTER A SYMBOL</label>
        <input
          id="customTicker"
          value={customTicker}
          onChange={(e) => setCustomTicker(e.target.value)}
          placeholder="e.g. MSFT or BRK-B"
          maxLength={15}
          autoCapitalize="characters"
        />
        <button type="submit">Load stock</button>
        {tickerError && <small className="ticker-error">{tickerError}</small>}
      </form>
      <div className="side-note">
        Live market data
        <br />
        GDELT headlines
        <br />
        Historical news datasets
      </div>
      <p className="muted bottom">
        RESEARCH PREVIEW
        <br />
        Not investment advice
      </p>
    </aside>
  );
}

// Shared panel header: icon, title, subtitle.
function Head({ icon, title, sub }) {
  return (
    <div className="panel-head">
      <div className="heading-icon">
        {icon}
        <span>
          <h2>{title}</h2>
          <p>{sub}</p>
        </span>
      </div>
    </div>
  );
}

function MetricTiles({ data }) {
  const history = data?.price_history || [];
  const tiles = [
    ["LATEST CLOSE", money(history.at(-1)?.Close), "5 day momentum " + pct(data?.momentum_5d)],
    ["5 DAY MODEL OUTLOOK", pct(data?.prediction), "Experimental direction model"],
    ["HISTORICAL MATCHES", (data?.matches || []).length, "Dated news events"],
    ["POSITIVE MATCH RATE", pct(data?.positive_match_rate), "Among closest events"],
  ];
  return (
    <section className="metrics">
      {tiles.map(([label, value, note]) => (
        <article key={label}>
          <small>{label}</small>
          <strong>{value}</strong>
          <span className="muted">{note}</span>
        </article>
      ))}
    </section>
  );
}

function PricePanel() {
  return (
    <article className="panel chart">
      <Head icon={<ChartNoAxesCombined />} title="Price performance" sub="Daily adjusted close · last 12 months" />
      <div id="priceChart" />
    </article>
  );
}

function ForecastPanel({ data }) {
  return (
    <article className="panel forecast">
      <Head icon={<Activity />} title="Prediction model" sub="Experimental: did not beat “always up” in testing" />
      <div className="gauge" id="gauge" />
      <div className="baseline">
        <span>Market only</span>
        <b>{pct(data?.market_baseline)}</b>
      </div>
      <p className="fine">Similarity weighted historical outcomes blended with the market baseline.</p>
    </article>
  );
}

// Past days whose headlines were most similar to today's, and what the stock did next.
function MatchesPanel({ matches }) {
  return (
    <article className="panel matches">
      <Head icon={<History />} title="Similar historical events" sub="Context and subsequent five-day return" />
      {matches.map((m) => (
        <div className="match" key={m.date}>
          <div className="headline">{m.headlines}</div>
          <div className="matchmeta">
            <span>
              {m.date} · similarity {(m.similarity * 100).toFixed(0)}%
            </span>
            <b className={m.actual_5d >= 0 ? "up" : "down"}>{pct(m.actual_5d)}</b>
          </div>
        </div>
      ))}
    </article>
  );
}

function ReplayPanel({ replay }) {
  const split = replay?.split;
  const subtitle = split
    ? `Train through ${split.train_through}; validate ${split.validation_start} to ${split.validation_end}; 2022+ held out`
    : "Loading chronological validation";
  const rows = [
    ["current_model", "CURRENT MODEL"],
    ["always_up_baseline", "ALWAYS UP (+1%)"],
  ];
  const current = replay?.metrics?.current_model;
  return (
    <article className="panel replay">
      <Head icon={<TrendingUp />} title="Validation · 5-day horizon" sub={subtitle} />
      <div className="replay-stats">
        {rows.map(([key, label]) => (
          <span key={key}>
            {label} MAE <b>{pct(replay?.metrics?.[key]?.mae)}</b> · direction{" "}
            <b>{pct(replay?.metrics?.[key]?.directional_accuracy)}</b>
          </span>
        ))}
      </div>
      <div className="fine">
        {replay?.architecture} · RMSE {pct(current?.rmse)} · {current?.samples || 0} validation forecasts, sampled
        every 5 trading days
      </div>
      <div id="replayChart" />
    </article>
  );
}

function HeadlinesPanel({ news, count }) {
  return (
    <section className="panel headlines">
      <Head icon={<Newspaper />} title="Latest financial coverage" sub="GDELT live feed + historical news datasets" />
      <div className="tag">{count || 0} ARTICLES</div>
      {news.slice(0, 12).map((n, i) => (
        <div className="newsitem" key={i}>
          <a href={n.url || "#"} target="_blank" rel="noreferrer">
            {n.title}
          </a>
          <small>
            {n.source}
            <br />
            {String(n.date).slice(0, 10)}
          </small>
        </div>
      ))}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Turbulence outlook: the main forecast (scripts/volatility.py)
// ---------------------------------------------------------------------------

// Plain-language tips, chosen by fixed rules from the model's drivers.
function turbulenceTips(t, ticker) {
  const { earnings_in_window: earningsSoon, days_to_earnings: days, vix_vs_normal: vix } = t.drivers;
  const tips = [];
  if (earningsSoon) {
    tips.push(
      `Earnings report in ${plural(days, "trading day")}. Stocks often jump sharply the day after a report, up or down. ` +
        `If you're about to buy, you might wait until after it. If you already own it, expect a big move so it doesn't catch you off guard.`,
    );
  } else if (days != null && days <= 15) {
    tips.push(`Next earnings report in ${days} trading days. Volatility usually rises around the report.`);
  }
  if (vix > 0.2) {
    tips.push(`Market-wide fear (VIX) is ${signedPct(vix)} above normal, so most stocks are likely to swing, not just ${ticker}.`);
  }
  tips.push(
    t.probability >= 0.5
      ? "Expect bigger daily swings than usual. Decide in advance what you'd do after a sharp drop instead of reacting in the moment."
      : "A calmer week is expected. Calm means smaller moves than usual, not that the price can't fall.",
  );
  return tips;
}

function earningsChip({ earnings_in_window: earningsSoon, days_to_earnings: days }) {
  if (earningsSoon) return `Earnings in ${plural(days, "trading day")}`;
  if (days == null) return "No earnings (index fund)";
  return `Next earnings in ${days} trading days`;
}

function Turbulence({ t, ticker }) {
  const [amount, setAmount] = useState(1000); // "If you have $___ invested"
  if (!t?.moves || !t?.drivers) return null; // outlook unavailable: hide the panel, keep the page working

  const turbulent = t.probability >= 0.5;
  const { drivers, moves } = t;
  const invested = Math.max(0, Number(amount) || 0);
  const expectedDaily = turbulent ? moves.turbulent_week_daily : moves.calm_week_daily;
  const otherDaily = turbulent ? moves.calm_week_daily : moves.turbulent_week_daily;

  return (
    <section className="panel turbulence">
      <Head
        icon={<Waves />}
        title={`Turbulence outlook · next ${t.horizon_days} trading days`}
        sub={`As of ${t.as_of} close · will ${ticker} move more than its usual amount, in either direction?`}
      />

      {/* Headline: label, probability and bar */}
      <div className="turb">
        <div className={"turb-label " + (turbulent ? "down" : "up")}>{turbulent ? "Turbulent" : "Calm"}</div>
        <div className="turb-prob">
          <b>{wholePct(t.probability)}</b> chance of an above-normal-volatility week
        </div>
        <div className="bar">
          <span style={{ width: `${t.probability * 100}%`, background: turbulent ? "#df5965" : "#19aa83" }} />
        </div>
      </div>

      {/* The three main drivers of the model */}
      <div className="chips">
        <span className="chip">VIX {signedPct(drivers.vix_vs_normal)} vs its 1-year norm</span>
        <span className="chip">Recent volatility {signedPct(drivers.recent_vol_vs_normal)} vs norm</span>
        <span className="chip">{earningsChip(drivers)}</span>
      </div>

      <div className="turb-grid">
        {/* Move sizes measured from the stock's own history, converted to dollars */}
        <div className="turb-box">
          <h3>What this means in dollars</h3>
          <p>
            {ticker} usually moves about <b>{plusMinusPct(moves.normal_daily)}</b> a day. Over the last 3 years it
            averaged <b>{plusMinusPct(moves.turbulent_week_daily)}</b> a day in turbulent weeks and{" "}
            <b>{plusMinusPct(moves.calm_week_daily)}</b> in calm weeks.
          </p>
          <label className="amount">
            If you have $
            <input type="number" min="0" step="100" value={amount} onChange={(e) => setAmount(e.target.value)} />{" "}
            invested:
          </label>
          <p>
            a typical day moves it about <b className={turbulent ? "down" : ""}>{plusMinusDollars(invested * expectedDaily)}</b>{" "}
            in a {turbulent ? "turbulent" : "calm"} week, against {plusMinusDollars(invested * otherDaily)} in a{" "}
            {turbulent ? "calm" : "turbulent"} one.
          </p>
        </div>

        <div className="turb-box">
          <h3>What this means for you</h3>
          <ul>
            {turbulenceTips(t, ticker).map((tip, i) => (
              <li key={i}>{tip}</li>
            ))}
          </ul>
        </div>
      </div>

      <p className="fine">
        Tested accuracy {wholePct(t.test_accuracy)} on held-out 2022–23 data ({wholePct(t.validation_accuracy)} in
        2015–21 validation) vs {wholePct(t.baseline_test_accuracy)} for “same as last week”. Predicts the size of
        moves, not their direction. Dollar figures are typical (root-mean-square) daily moves from past data, not
        guarantees. Educational prototype, not financial advice.
      </p>
    </section>
  );
}

createRoot(document.getElementById("root")).render(<App />);
