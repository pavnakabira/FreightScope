"""
FreightScope backend — SIH26006
Serves real freight forecasts + vessel-port feasibility as JSON for the dashboard.

Run:
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8000
Then open: http://localhost:8000/docs  (interactive API)

Endpoints:
    GET /health
    GET /indices                      -> latest Baltic index values + source used
    GET /forecast?index=BPI&days=30   -> historical + Prophet forecast w/ confidence band
    GET /recommend?origin=...&dest=...&cargo=...  -> full recommendation payload
    GET /ports                        -> verified port constraint table
"""

from __future__ import annotations
import io, datetime as dt
from functools import lru_cache
from typing import Optional

import pandas as pd
import numpy as np
import requests
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="FreightScope API", version="1.0")
# allow the dashboard (any origin) to call this during the hackathon
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ----------------------------------------------------------------------------
# 1. VERIFIED PORT + VESSEL REFERENCE DATA (from official sources, Sep 2026)
# ----------------------------------------------------------------------------
PORTS = {
    "Paradip":       {"state": "Odisha",         "loa": 300, "beam": 46.0,  "draft": 16.5, "cls": "Capesize", "cargo": "Thermal/coking coal, iron ore", "verified": True, "tidal_bonus": 0.0, "tide_note": "deep-water, minimal tidal dependence", "advisory": "Mechanised coal berths; ample air draft under loaders."},
    "Visakhapatnam": {"state": "Andhra Pradesh", "loa": 390, "beam": 50.0,  "draft": 18.1, "cls": "Capesize", "cargo": "Coking/thermal coal, iron ore", "verified": True, "tidal_bonus": 0.5, "tide_note": "outer harbour; +0.5m on rising tide", "advisory": "Use outer-harbour berths for Capesize; inner harbour is draft-limited."},
    "Gangavaram":    {"state": "Andhra Pradesh", "loa": 325, "beam": 65.0,  "draft": 18.0, "cls": "Capesize", "cargo": "Coal, iron ore, bauxite",       "verified": True, "tidal_bonus": 0.0, "tide_note": "deep-water", "advisory": "One of India's deepest ports; fully-laden Capesize capable."},
    "Dhamra":        {"state": "Odisha",         "loa": 300, "beam": 47.0,  "draft": 17.5, "cls": "Capesize", "cargo": "Coal, iron ore, limestone",    "verified": True, "tidal_bonus": 0.0, "tide_note": "deep-water", "advisory": "Deep-water; check monsoon-season swell for berthing windows (Jun–Sep)."},
    "Gopalpur":      {"state": "Odisha",         "loa": 240, "beam": 42.0,  "draft": 14.5, "cls": "Panamax",  "cargo": "Coal, limestone, ilmenite",    "verified": False, "tidal_bonus": 0.0, "tide_note": "figures indicative", "advisory": "Developing port; confirm current draft and seasonal restrictions before fixing."},
    "Haldia":        {"state": "West Bengal",    "loa": 240, "beam": 32.26, "draft": 11.0, "cls": "Supramax", "cargo": "Coking/thermal coal, steel",   "verified": True, "tidal_bonus": 2.5, "tide_note": "river port: fresh water, strongly tide-bound; up to +2.5m only in a high-tide window", "advisory": "Fresh-water draft; night-navigation limits for >8.5m draft & >230m LOA; monsoon river-draft variation (Jul–Oct)."},
}

VCLASS = {
    "Handysize": {"draft": 10.0, "beam": 27.0,  "loa": 180, "usd_per_t": 23.4, "idx": "BHSI", "max_cargo": 40000},
    "Supramax":  {"draft": 12.5, "beam": 32.0,  "loa": 200, "usd_per_t": 20.1, "idx": "BSI",  "max_cargo": 65000},
    "Panamax":   {"draft": 14.3, "beam": 32.3,  "loa": 230, "usd_per_t": 18.3, "idx": "BPI",  "max_cargo": 85000},
    "Capesize":  {"draft": 18.0, "beam": 47.0,  "loa": 300, "usd_per_t": 17.8, "idx": "BCI",  "max_cargo": 200000},
}
CLASS_ORDER = ["Capesize", "Panamax", "Supramax", "Handysize"]
CAP_RANK = {"Handysize": 1, "Supramax": 2, "Panamax": 3, "Capesize": 4}

# origin regions: nautical miles to East Coast + max loadable class (load-port constraint)
ORIGINS = {
    "Australia — Hay Point":   {"nm": 10800, "max_class": "Capesize", "load_rate": 55000, "freight_diff": 1.00, "load_note": "Deep-water coal terminal; Capesize-capable, fast loading."},
    "USA — Hampton Roads":     {"nm": 12500, "max_class": "Capesize", "load_rate": 45000, "freight_diff": 1.08, "load_note": "Longest haul to East Coast India; Suez/Cape routing affects rate."},
    "Mozambique — Beira":      {"nm": 4200,  "max_class": "Panamax",  "load_rate": 25000, "freight_diff": 0.95, "load_note": "Draft-limited load port; Panamax max, slower loading."},
    "Indonesia — Kalimantan":  {"nm": 3900,  "max_class": "Supramax", "load_rate": 30000, "freight_diff": 0.90, "load_note": "Short haul; anchorage/transhipment loading, Supramax typical."},
    "Russia — Vostochny":      {"nm": 7600,  "max_class": "Capesize", "load_rate": 40000, "freight_diff": 1.03, "load_note": "Route/geopolitics-sensitive; check sanctions & payment terms."},
}

# Per-lane distance adjustments: destination ports differ in distance from each
# origin. Multiplier applied to the base origin distance (representative).
LANE_DEST_FACTOR = {
    "Paradip": 1.00, "Visakhapatnam": 1.02, "Gangavaram": 1.02,
    "Dhamra": 0.99, "Gopalpur": 1.01, "Haldia": 1.05,   # Haldia further up the Hooghly
}

def lane_distance(origin_nm: float, dest: str) -> float:
    return round(origin_nm * LANE_DEST_FACTOR.get(dest, 1.0), 0)

# ----------------------------------------------------------------------------
# 2. DATA SOURCES
# ----------------------------------------------------------------------------
# The true Baltic Dry Index is licensed by the Baltic Exchange and has no free
# live API. So the data layer uses three tiers, in order:
#   1. FRED (live, free, citable) — real daily freight-driver series.
#   2. A historical BDI CSV you drop in ./data/ — real Baltic values for
#      training/backtesting the models.
#   3. Synthetic — deterministic fallback, clearly labelled, so the API and
#      demo never break offline.
#
# FRED series actually used (all confirmed reachable, free, daily):
#   DCOILBRENTEU — Brent crude (USD/bbl): bunker-fuel cost, a real freight driver.
#   DCOILWTICO   — WTI crude (USD/bbl): alternative fuel benchmark.
# These are proxies for freight-cost pressure, NOT the BDI itself. Labelled as such.
import os

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
FRED_SERIES = {          # our index code -> (FRED id, human label)
    "BRENT": ("DCOILBRENTEU", "Brent crude (fuel-cost proxy)"),
    "WTI":   ("DCOILWTICO",   "WTI crude (fuel-cost proxy)"),
}
# Baltic codes map to a local historical CSV if present, else synthetic.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# CSV filename expected per Baltic code (drop real history here). Column names
# are auto-detected: a date column + a value column.
BALTIC_CSV = {
    "BDI": "bdi_history.csv", "BCI": "bci_history.csv", "BPI": "bpi_history.csv",
    "BSI": "bsi_history.csv", "BHSI": "bhsi_history.csv",
}


def _fetch_fred(series_id: str) -> Optional[pd.DataFrame]:
    """Fetch a daily series from FRED as df[Date, Close]. None on failure."""
    try:
        r = requests.get(FRED_CSV.format(sid=series_id), timeout=20)
        if r.status_code != 200 or "observation_date" not in r.text[:60]:
            return None
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ["Date", "Close"]                       # 2nd col is the series
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")  # FRED uses '.' for gaps
        df = df.dropna().reset_index(drop=True)
        return df if len(df) > 60 else None
    except Exception:
        return None


# BDRY = Breakwave Dry Bulk Shipping ETF. It holds Baltic freight futures, so it
# tracks the Baltic Dry Index closely and is free/live via Yahoo Finance. We use
# it as the real Baltic signal, scaled to each sub-index's typical level.
BDRY_SCALE = {  # ETF price (~$15) -> representative index level
    "BDI": 130.0, "BCI": 210.0, "BPI": 120.0, "BSI": 95.0, "BHSI": 50.0,
}

def _fetch_bdry(period: str = "2y") -> Optional[pd.DataFrame]:
    """Fetch BDRY ETF daily closes via yfinance as df[Date, Close]. None on failure.
    Yahoo sometimes rejects longer windows without a crumb, so try shorter ones."""
    try:
        import yfinance as yf
    except Exception:
        return None
    for per in (period, "1y", "6mo"):
        try:
            raw = yf.download("BDRY", period=per, progress=False, auto_adjust=True)
            if raw is None or raw.empty:
                continue
            close = raw["Close"]
            if hasattr(close, "columns"):        # multiindex -> take the BDRY column
                close = close.iloc[:, 0]
            df = pd.DataFrame({"Date": pd.to_datetime(close.index), "Close": close.values})
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            df = df.dropna().reset_index(drop=True)
            if len(df) > 60:
                return df
        except Exception:
            continue
    return None


def _load_csv(path: str) -> Optional[pd.DataFrame]:
    """Load a local historical CSV; auto-detect date + value columns."""
    try:
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        date_col = next((cols[c] for c in cols if "date" in c or "observation" in c), df.columns[0])
        val_col  = next((cols[c] for c in cols
                         if any(k in c for k in ("close", "value", "price", "bdi", "index"))),
                        df.columns[-1])
        out = df[[date_col, val_col]].copy()
        out.columns = ["Date", "Close"]
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
        out = out.dropna().sort_values("Date").reset_index(drop=True)
        return out if len(out) > 30 else None
    except Exception:
        return None


def _synthetic_series(base: float, n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Deterministic synthetic history so the API always works offline/in demo."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=dt.date.today(), periods=n)
    m = len(dates)                     # bdate_range can return n±1; align to actual
    steps = rng.normal(0, base * 0.012, m)
    season = base * 0.06 * np.sin(np.linspace(0, 6 * np.pi, m))
    vals = np.clip(base + np.cumsum(steps) * 0.4 + season, base * 0.4, base * 1.9)
    return pd.DataFrame({"Date": dates, "Close": np.round(vals, 2)})


@lru_cache(maxsize=16)
def get_series(index_code: str) -> tuple:
    """
    Returns (df[Date,Close], source_label). Resolution order:
      FRENT/WTI  -> live FRED
      BDI/BCI/... -> local historical CSV in ./data/ if present
      anything   -> synthetic fallback (labelled)
    """
    code = index_code.upper()

    # 1. live FRED series
    if code in FRED_SERIES:
        sid, label = FRED_SERIES[code]
        df = _fetch_fred(sid)
        if df is not None:
            return (df, f"FRED {sid} (live) — {label}")
        base = {"BRENT": 75, "WTI": 72}.get(code, 75)
        return (_synthetic_series(base, seed=abs(hash(code)) % 1000),
                f"synthetic ({code} — FRED unreachable)")

    # 2. Baltic code -> live BDRY ETF (tracks BDI), else local CSV, else synthetic
    if code in BALTIC_CSV:
        # 2a. live BDRY, scaled to this sub-index's typical level
        bdry = _fetch_bdry()
        if bdry is not None:
            df = bdry.copy()
            ref = df["Close"].tail(20).mean()
            target = BDRY_SCALE.get(code, 130.0)
            df["Close"] = (df["Close"] / ref * target).round(1)   # scale ETF -> index level
            note = "" if code == "BDI" else f" scaled to {code}"
            return (df, f"BDRY ETF (live, tracks BDI){note}")
        # 2b. local historical CSV
        path = os.path.join(DATA_DIR, BALTIC_CSV[code])
        if os.path.exists(path):
            df = _load_csv(path)
            if df is not None:
                return (df, f"historical CSV ({BALTIC_CSV[code]}) — real {code}, not live")
        # 2c. synthetic
        base = {"BCI": 4400, "BPI": 2200, "BSI": 1650, "BHSI": 880, "BDI": 2776}.get(code, 2200)
        return (_synthetic_series(base, seed=abs(hash(code)) % 1000),
                f"synthetic ({code} — BDRY & CSV unavailable)")

    # 3. unknown code
    return (_synthetic_series(2000, seed=1), f"synthetic (unknown code {code})")


# ----------------------------------------------------------------------------
# 3. FORECAST  (Prophet with graceful fallback to simple trend)
# ----------------------------------------------------------------------------
def prophet_forecast(df: pd.DataFrame, days: int = 30) -> dict:
    hist = df.tail(180).copy()  # last ~9 months for speed
    try:
        from prophet import Prophet
        pdf = hist.rename(columns={"Date": "ds", "Close": "y"})
        # yearly seasonality off: we feed ~180 days, too little to identify it
        m = Prophet(daily_seasonality=False, weekly_seasonality=True,
                    yearly_seasonality=False, interval_width=0.8)
        m.fit(pdf)
        future = m.make_future_dataframe(periods=days, freq="B")
        fc = m.predict(future)
        tail = fc.tail(days)
        return {
            "engine": "Prophet",
            "hist_dates":  hist["Date"].dt.strftime("%Y-%m-%d").tolist(),
            "hist_values": hist["Close"].round(1).tolist(),
            "fc_dates":  tail["ds"].dt.strftime("%Y-%m-%d").tolist(),
            "fc_values": tail["yhat"].round(1).tolist(),
            "fc_lower":  tail["yhat_lower"].round(1).tolist(),
            "fc_upper":  tail["yhat_upper"].round(1).tolist(),
        }
    except Exception as e:
        # simple linear-trend fallback so the endpoint never fails
        y = hist["Close"].values
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        fut_x = np.arange(len(y), len(y) + days)
        yhat = slope * fut_x + intercept
        resid = float(np.std(y - (slope * x + intercept)))
        fut_dates = pd.bdate_range(start=hist["Date"].iloc[-1] + pd.Timedelta(days=1), periods=days)
        return {
            "engine": f"linear-trend fallback ({type(e).__name__})",
            "hist_dates":  hist["Date"].dt.strftime("%Y-%m-%d").tolist(),
            "hist_values": hist["Close"].round(1).tolist(),
            "fc_dates":  fut_dates.strftime("%Y-%m-%d").tolist(),
            "fc_values": np.round(yhat, 1).tolist(),
            "fc_lower":  np.round(yhat - 1.28 * resid, 1).tolist(),
            "fc_upper":  np.round(yhat + 1.28 * resid, 1).tolist(),
        }


# ----------------------------------------------------------------------------
# 4. FEASIBILITY + RECOMMENDATION LOGIC
# ----------------------------------------------------------------------------
def feasible_classes(dest: str, origin_max_class: str, parcel_mt: int = 0) -> list:
    p = PORTS[dest]
    load_cap = CAP_RANK.get(origin_max_class, 4)
    tidal = p.get("tidal_bonus", 0.0)
    draft_all_tide = p["draft"]                 # available all-tide
    draft_high_tide = p["draft"] + tidal        # available only in the high-tide window
    out = []
    for c in CLASS_ORDER:
        vc = VCLASS[c]
        fits_beam = vc["beam"] <= p["beam"]
        fits_loa = vc["loa"] <= p["loa"]
        fits_load = CAP_RANK[c] <= load_cap
        fits_all_tide = vc["draft"] <= draft_all_tide
        fits_high_tide = vc["draft"] <= draft_high_tide
        # parcel must fit the hold; only checked if a parcel size is given
        fits_hold = (parcel_mt == 0) or (parcel_mt <= vc["max_cargo"])
        tide_restricted = (not fits_all_tide) and fits_high_tide

        feasible = fits_beam and fits_loa and fits_load and fits_high_tide and fits_hold

        if not fits_beam:
            reason = f"beam {vc['beam']}m > {p['beam']}m port limit"
        elif not fits_loa:
            reason = f"LOA {vc['loa']}m > {p['loa']}m port limit"
        elif not fits_high_tide:
            reason = f"draft {vc['draft']}m > {round(draft_high_tide,1)}m even at high tide"
        elif not fits_load:
            reason = f"exceeds load-port max ({origin_max_class})"
        elif not fits_hold:
            reason = f"parcel {parcel_mt:,}t > {vc['max_cargo']:,}t hold capacity — split or size up"
        elif tide_restricted:
            reason = f"draft {vc['draft']}m OK only in high-tide window (+{tidal}m); plan tidal berthing"
        else:
            reason = f"draft {vc['draft']}m <= {draft_all_tide}m all-tide · {vc['idx']}"

        out.append({
            "cls": c, "usd_per_t": vc["usd_per_t"], "idx": vc["idx"],
            "feasible": feasible,
            "tide_restricted": tide_restricted and feasible,
            "reason": reason,
        })
    return out


def best_entry_window(fc: dict) -> dict:
    """Find the lowest-forecast day within the horizon = best time to charter,
    with a probability-of-dip estimate from the forecast confidence band."""
    vals = fc["fc_values"]
    if not vals:
        return {"day_index": 0, "date": None, "delta_pct": 0.0}
    today = fc["hist_values"][-1] if fc["hist_values"] else vals[0]
    i_min = int(np.argmin(vals))
    delta = (vals[i_min] - today) / today * 100 if today else 0.0

    # Probability the rate dips at least `dip_pct` below today within a near horizon,
    # approximated from the 80% confidence band (lower/upper) treated as ~1.28 sigma.
    lower = fc.get("fc_lower", []); upper = fc.get("fc_upper", [])
    prob_dip = None; dip_pct = 3.0; horizon = min(10, len(vals))
    if lower and upper and today:
        target = today * (1 - dip_pct / 100)
        best_p = 0.0
        for k in range(horizon):
            mu = vals[k]
            half = (upper[k] - lower[k]) / 2.0 if k < len(upper) and k < len(lower) else 0
            sigma = half / 1.28 if half > 0 else max(1e-6, abs(mu) * 0.02)
            # P(rate <= target) under Normal(mu, sigma)
            from math import erf, sqrt
            z = (target - mu) / (sigma * sqrt(2))
            p = 0.5 * (1 + erf(z))
            best_p = max(best_p, p)
        prob_dip = round(best_p * 100, 0)

    return {"day_index": i_min, "date": fc["fc_dates"][i_min],
            "forecast_low": vals[i_min], "today": today, "delta_pct": round(delta, 1),
            "prob_dip_pct": prob_dip, "dip_threshold_pct": dip_pct, "dip_horizon_days": horizon}


# ----------------------------------------------------------------------------
# 5. ENDPOINTS
# ----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "time": dt.datetime.utcnow().isoformat()}


@app.get("/")
def dashboard():
    """Serve the live dashboard if dashboard_live.html sits next to app.py."""
    from fastapi.responses import FileResponse, HTMLResponse
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "dashboard_live.html")
    if os.path.exists(path):
        return FileResponse(path)
    return HTMLResponse("<h3>FreightScope API is running.</h3>"
                        "<p>Put dashboard_live.html next to app.py to see the dashboard here, "
                        "or open <a href='/docs'>/docs</a>.</p>")


@app.get("/coa")
def coa(
    origin: str = Query("Australia — Hay Point"),
    dest: str = Query("Paradip"),
    cargo: str = Query("Coking coal"),
    total_mt: int = Query(450000, ge=50000, le=5000000),
    months: int = Query(3, ge=1, le=12),
    parcel_mt: int = Query(75000, ge=10000, le=220000),
    bunker_usd_per_t: float = Query(600.0),
):
    """Contract-of-Affreightment programme planner: tonnage target across voyages."""
    from coa_planner import plan_programme
    origin_info = ORIGINS.get(origin, {"nm": 8000, "max_class": "Capesize"})
    feas = feasible_classes(dest, origin_info["max_class"], parcel_mt)
    eligible = [f for f in feas if f["feasible"]]
    if not eligible:
        return {"error": f"No feasible vessel for {origin}→{dest} at {parcel_mt:,}t parcel.",
                "hint": "Reduce parcel size or change ports."}
    pick = min(eligible, key=lambda f: VCLASS[f["cls"]]["usd_per_t"])["cls"]
    idx = VCLASS[pick]["idx"]
    df, src = get_series(idx)
    if src.startswith("synthetic"):
        df, src = get_series("BRENT")
    fc = prophet_forecast(df, 45)
    return plan_programme(pick, total_mt, months, lane_distance(origin_info["nm"], dest), cargo,
                          VCLASS[pick]["usd_per_t"], bunker_usd_per_t,
                          fc.get("fc_dates", []), fc.get("fc_values", []), parcel_mt)


@app.get("/forecast_accuracy")
def forecast_accuracy(index: str = Query("BDI")):
    """Realised forecast-vs-actual accuracy, updated as forecast dates mature."""
    from forecast_tracker import score
    df, _ = get_series(index)
    actual = [{"Date": d.strftime("%Y-%m-%d"), "Close": float(c)}
              for d, c in zip(df["Date"], df["Close"])]
    return score(index, actual)


@app.get("/proxy_quality")
def proxy_quality_endpoint():
    """BDRY-vs-BDI correlation and tracking error — quantifies the freight proxy."""
    from proxy_quality import proxy_quality, load_bdi_reference
    bdry, _ = get_series("BDI")   # this is the BDRY-derived series
    bdi_ref = load_bdi_reference()
    return proxy_quality(bdry, bdi_ref)


@app.get("/voyage")
def voyage(
    vessel_class: str = Query("Panamax"),
    distance_nm: float = Query(10800),
    cargo_mt: float = Query(75000),
    freight_rate_usd_per_t: float = Query(18.3),
    bunker_usd_per_t: float = Query(600.0),
    cargo_type: str = Query("Coking coal"),
):
    """Standalone, fully itemised voyage-cost estimate — the transparent basis for $/tonne."""
    from voyage_cost import estimate_voyage
    return estimate_voyage(vessel_class, distance_nm, cargo_mt,
                           freight_rate_usd_per_t, bunker_usd_per_t, cargo_type)


@app.get("/ports")
def ports():
    return {"ports": PORTS, "vessel_classes": VCLASS, "origins": ORIGINS}


_CONGESTION_CACHE = {"ts": 0, "days": 0, "result": None}

@app.get("/congestion")
def congestion(days: int = Query(10, ge=3, le=30)):
    """
    Real Paradip berth-queue congestion from Daily Traffic Reports.
    Cached for 30 minutes so the tab is instant after first load; falls back to
    an empty series (with a note) if the port site is unreachable.
    """
    import time
    now = time.time()
    # serve cache if fresh (<30 min) and same/greater coverage
    if (_CONGESTION_CACHE["result"] and now - _CONGESTION_CACHE["ts"] < 1800
            and _CONGESTION_CACHE["days"] >= days):
        return _CONGESTION_CACHE["result"]
    try:
        from paradip_dtr_scraper import build_congestion_series
        rows = build_congestion_series(days=days)
        if rows:
            latest = rows[-1]
            avg = round(sum(r["congestion_index"] for r in rows) / len(rows), 3)
            level = ("high" if latest["congestion_index"] > 1.2
                     else "moderate" if latest["congestion_index"] > 0.7 else "low")
            result = {"source": "Paradip DTR (live)", "series": rows,
                      "latest": latest, "avg_congestion": avg, "level": level}
            _CONGESTION_CACHE.update(ts=now, days=days, result=result)
            return result
        return {"source": "Paradip DTR unreachable", "series": [], "latest": None,
                "avg_congestion": None, "level": "unknown",
                "note": "Port site not reachable right now; try again shortly."}
    except Exception as e:
        return {"source": f"scraper error ({type(e).__name__})", "series": [],
                "latest": None, "avg_congestion": None, "level": "unknown"}


@app.get("/indices")
def indices():
    # BRENT/WTI are LIVE from FRED; Baltic codes come from local CSV or synthetic.
    out = {}
    for code in ["BRENT", "WTI", "BDI", "BCI", "BPI", "BSI", "BHSI"]:
        df, src = get_series(code)
        out[code] = {"latest": float(df["Close"].iloc[-1]),
                     "as_of": df["Date"].iloc[-1].strftime("%Y-%m-%d"),
                     "live": src.startswith(("FRED", "BDRY")),
                     "source": src}
    return out


@app.get("/compare")
def compare(index: str = Query("BPI"), horizon: int = Query(7, ge=3, le=14),
            folds: int = Query(6, ge=3, le=10)):
    """
    Walk-forward backtest: Naive vs Prophet vs XGBoost on the same held-out days.
    Returns MAE/RMSE/MAPE per model + skill vs naive + last-fold predictions.
    This is the evidence that the ML earns its place.
    """
    df, src = get_series(index)
    try:
        from forecasting import compare_models
        rep = compare_models(df, horizon=horizon, folds=folds)
        rep["index"] = index.upper()
        rep["source"] = src
        return rep
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "index": index.upper(), "source": src}


@app.get("/forecast")
def forecast(index: str = Query("BPI"), days: int = Query(30, ge=7, le=90)):
    df, src = get_series(index)
    fc = prophet_forecast(df, days)
    fc["index"] = index.upper()
    fc["source"] = src
    fc["entry_window"] = best_entry_window(fc)
    # Add an XGBoost forecast line alongside Prophet's, so the chart shows both
    # rather than leading solely with Prophet. Honest framing: Prophet supplies the
    # confidence band; XGBoost is the stronger short-horizon model on our backtest.
    try:
        from forecasting import make_features, _fit_predict_xgb, FEATURE_COLS
        feat = make_features(df)
        xgb_vals = _fit_predict_xgb(feat, FEATURE_COLS, days)
        fc["xgb_values"] = [round(float(v), 1) for v in xgb_vals]
        fc["model_note"] = ("Prophet (band) shown with XGBoost overlay. On the "
                            "walk-forward backtest XGBoost leads Prophet at short "
                            "horizons; see /compare for the scored comparison.")
    except Exception:
        fc["xgb_values"] = []
    # log this forecast for later accuracy scoring (best-effort)
    try:
        from forecast_tracker import log_forecast
        log_forecast(index, fc.get("fc_dates", []), fc.get("fc_values", []))
    except Exception:
        pass
    return fc


@app.get("/recommend")
def recommend(
    origin: str = Query("Australia — Hay Point"),
    dest: str = Query("Paradip"),
    cargo: str = Query("Coking coal"),
    horizon_days: int = Query(60, ge=7, le=90),
    parcel_mt: int = Query(75000, ge=5000, le=220000),
    bunker_usd_per_t: float = Query(600.0, ge=200, le=1200),
):
    from voyage_cost import estimate_voyage
    origin_info = ORIGINS.get(origin, {"nm": 8000, "max_class": "Capesize"})
    nm = lane_distance(origin_info["nm"], dest)
    feas = feasible_classes(dest, origin_info["max_class"], parcel_mt)

    # Freight rate per class, ANCHORED TO THE LIVE MARKET. We take each class's
    # live index level (via its Baltic sub-index) and scale the representative
    # $/t reference rate by the live level relative to a reference baseline, so the
    # voyage cost moves with the real market instead of being a fixed constant.
    # Reference baselines are the index levels at which the reference $/t rates hold.
    INDEX_BASELINE = {"BCI": 208.0, "BPI": 119.0, "BSI": 94.0, "BHSI": 50.0}
    def live_level(cls):
        code = VCLASS[cls]["idx"]
        try:
            df, _ = get_series(code)
            return float(df["Close"].iloc[-1])
        except Exception:
            return None
    def freight_rate_for(cls):
        base_rate = VCLASS[cls]["usd_per_t"]
        code = VCLASS[cls]["idx"]
        lvl = live_level(cls); baseline = INDEX_BASELINE.get(code)
        if lvl and baseline and baseline > 0:
            ratio = max(0.5, min(2.0, lvl / baseline))   # clamp to avoid wild scaling
            return round(base_rate * ratio, 2)
        return base_rate

    # Compute a real voyage cost for each FEASIBLE class and attach the breakdown.
    for f in feas:
        if f["feasible"]:
            est = estimate_voyage(
                f["cls"], distance_nm=nm, cargo_mt=parcel_mt,
                freight_rate_usd_per_t=freight_rate_for(f["cls"]),
                bunker_price_usd_per_t=bunker_usd_per_t, cargo_type=cargo,
            )
            f["voyage"] = est
            f["usd_per_t_modelled"] = est["result"]["usd_per_tonne"]
            f["tce_per_day"] = est["result"]["tce_per_day_usd"]

    eligible = [f for f in feas if f["feasible"]]
    # pick the lowest MODELLED cost per tonne (real voyage economics, not a lookup)
    pick = min(eligible, key=lambda f: f["usd_per_t_modelled"])["cls"] if eligible else None

    # --- "What if I split?" analysis ---
    # For any class infeasible ONLY because the parcel exceeds its hold capacity,
    # compute the cost of splitting the parcel into the minimum number of shipments
    # that fit, and compare it to the recommended single-ship option.
    split_options = []
    import math
    base_feas = feasible_classes(dest, origin_info["max_class"], 0)
    for f in feas:
        if f.get("feasible"):
            continue
        reason = f.get("reason", "")
        if "hold capacity" not in reason:
            continue
        cls = f["cls"]
        cap = VCLASS[cls]["max_cargo"]
        fits_otherwise = next((b["feasible"] for b in base_feas if b["cls"] == cls), False)
        if not fits_otherwise:
            continue
        try:
            n_ship = max(2, math.ceil(parcel_mt / cap))
            per = round(parcel_mt / n_ship)
            est = estimate_voyage(cls, distance_nm=nm, cargo_mt=per,
                                  freight_rate_usd_per_t=freight_rate_for(cls),
                                  bunker_price_usd_per_t=bunker_usd_per_t, cargo_type=cargo)
            total = est["result"]["total_charterer_cost_usd"] * n_ship
            split_options.append({
                "cls": cls, "n_shipments": n_ship, "per_shipment_mt": per,
                "usd_per_t": round(total / parcel_mt, 2),
                "total_cost_usd": round(total, 0),
                "single_ship_usd_per_t": None,
            })
        except Exception:
            continue

    result = {
        "origin": origin, "dest": dest, "cargo": cargo, "parcel_mt": parcel_mt,
        "distance_nm": nm,
        "transit_days": round(nm / 13 / 24, 1),
        "vessels": feas,
        "recommended_class": pick,
        "port": PORTS[dest],
        "bunker_usd_per_t": bunker_usd_per_t,
        "load_note": origin_info.get("load_note"),
    }
    # attach single-ship comparison + verdict to each split option
    if pick and split_options:
        pick_upt = next(f["usd_per_t_modelled"] for f in eligible if f["cls"] == pick)
        for so in split_options:
            so["single_ship_class"] = pick
            so["single_ship_usd_per_t"] = pick_upt
            so["cheaper"] = so["usd_per_t"] < pick_upt
            diff = round((so["usd_per_t"] - pick_upt) * parcel_mt, 0)
            so["vs_single_usd"] = diff   # +ve = split costs more
    result["split_options"] = split_options
    if pick:
        pick_f = next(f for f in eligible if f["cls"] == pick)
        idx = VCLASS[pick]["idx"]
        df, src = get_series(idx)
        if src.startswith("synthetic"):
            df, src = get_series("BRENT")
            idx = f"{idx} (proxy: live Brent — BDRY unavailable)"
        fc = prophet_forecast(df, min(horizon_days, 45))
        win = best_entry_window(fc)
        result["forecast"] = {"index": idx, "engine": fc["engine"], "source": src,
                              "entry_window": win}
        result["cost"] = pick_f["voyage"]          # full itemised breakdown for the pick
        upt = pick_f["usd_per_t_modelled"]
        tce = pick_f["tce_per_day"]

        # --- Quantified saving: recommended entry window vs. chartering spot today ---
        # The forecast gives today's level and the forecast-low level for this index.
        # We translate that percentage move into the freight rate and re-run the
        # voyage cost, so the saving is a real landed-cost saving, not an index delta.
        today_lvl = win.get("today")
        low_lvl = win.get("forecast_low")
        saving = None
        if today_lvl and low_lvl and today_lvl > 0:
            base_rate = freight_rate_for(pick)                     # rate at "today"
            ratio = low_lvl / today_lvl                            # forecast move
            window_rate = round(base_rate * ratio, 2)             # rate at best window
            cost_today = estimate_voyage(pick, nm, parcel_mt, base_rate,
                                         bunker_usd_per_t, cargo)["result"]
            cost_window = estimate_voyage(pick, nm, parcel_mt, window_rate,
                                          bunker_usd_per_t, cargo)["result"]
            per_t = round(cost_today["usd_per_tonne"] - cost_window["usd_per_tonne"], 2)
            total = round(cost_today["total_charterer_cost_usd"] - cost_window["total_charterer_cost_usd"], 0)
            saving = {
                "spot_today_usd_per_t": cost_today["usd_per_tonne"],
                "recommended_window_usd_per_t": cost_window["usd_per_tonne"],
                "saving_usd_per_t": per_t,
                "saving_total_usd": total,
                "saving_pct": round((per_t / cost_today["usd_per_tonne"] * 100), 1) if cost_today["usd_per_tonne"] else 0,
                "window_date": win.get("date"),
                "basis": "recommended entry window vs. spot-today, priced through the voyage-cost model",
            }
        result["saving"] = saving

        # --- Demurrage risk from live congestion (Paradip only for now) ---
        from demurrage import demurrage_estimate
        cong_idx = None
        if dest == "Paradip":
            try:
                from paradip_dtr_scraper import build_congestion_series
                rows = build_congestion_series(days=7)
                if rows:
                    cong_idx = sum(r["congestion_index"] for r in rows) / len(rows)
            except Exception:
                cong_idx = None
        result["demurrage"] = demurrage_estimate(pick, cong_idx) if cong_idx is not None else None

        # --- Idle & repositioning analysis (deliverable c) ---
        from idle_repositioning import idle_analysis
        # nearest alternative load region distance (rough, from origin set); use a
        # representative 1800nm if unknown, so the economics are illustrated.
        result["idle"] = idle_analysis(pick, fc.get("fc_values", []), fc.get("fc_dates", []),
                                       reposition_nm=1800, bunker_price=bunker_usd_per_t)

        save_txt = ""
        if saving and saving["saving_total_usd"] > 0:
            save_txt = (f" Acting on the window instead of spot today saves "
                        f"~${saving['saving_usd_per_t']}/t (~${saving['saving_total_usd']:,.0f} on this parcel).")
        prob_txt = ""
        pd = win.get("prob_dip_pct")
        if pd is not None:
            prob_txt = (f" ~{pd:.0f}% chance of a {win.get('dip_threshold_pct',3):.0f}%+ dip "
                        f"within {win.get('dip_horizon_days',10)} days.")
        result["verdict"] = (
            f"Charter {pick} around {win.get('date','the low')} "
            f"({win.get('delta_pct',0)}% vs today) — cheapest feasible class at "
            f"~${upt}/t all-in (TCE ~${tce:,}/day).{save_txt}{prob_txt}"
        )
    else:
        result["verdict"] = (f"No standard vessel class fits both {origin.split('—')[0].strip()} "
                             f"and {dest}. Consider transhipment or splitting the parcel.")
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
