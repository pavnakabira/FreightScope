"""
forecast_tracker.py — SIH26006
Forecast-vs-actual tracking: logs each day's forecast and, when the actual
arrives, scores it. Builds justified trust and monitors model performance over
time — directly answering "can I trust the timing advice?".

Storage: a small JSON log in ./data/forecast_log.json. Each entry:
  {date_made, index, horizon_date, predicted, actual (filled later)}
On read, it aligns any entries whose horizon_date now has an actual value from
the live series and reports realised MAE/MAPE.
"""
from __future__ import annotations
import os, json, datetime as dt

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LOG_PATH = os.path.join(DATA_DIR, "forecast_log.json")


def _load():
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def _save(rows):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(rows, f, indent=2)


def log_forecast(index: str, horizon_dates: list[str], predicted: list[float]):
    """Record today's forecast for later scoring (dedup by date_made+horizon)."""
    rows = _load()
    today = dt.date.today().isoformat()
    seen = {(r["date_made"], r["horizon_date"], r["index"]) for r in rows}
    for hd, pv in zip(horizon_dates, predicted):
        key = (today, hd, index.upper())
        if key not in seen:
            rows.append({"date_made": today, "index": index.upper(),
                         "horizon_date": hd, "predicted": round(float(pv), 2),
                         "actual": None})
    _save(rows)
    return len(rows)


def score(index: str, actual_series: list[dict]) -> dict:
    """
    actual_series: [{Date: 'YYYY-MM-DD', Close: value}, ...] from the live series.
    Fills in actuals for matured forecasts and returns realised accuracy.
    """
    rows = _load()
    actual_map = {a["Date"]: a["Close"] for a in actual_series}
    matured = []
    for r in rows:
        if r["index"] != index.upper():
            continue
        if r["actual"] is None and r["horizon_date"] in actual_map:
            r["actual"] = round(float(actual_map[r["horizon_date"]]), 2)
        if r["actual"] is not None:
            matured.append(r)
    _save(rows)
    if not matured:
        return {"scored_points": 0,
                "note": "No matured forecasts yet — accuracy will populate as forecast dates pass. Log grows each run."}
    errs = [abs(r["predicted"] - r["actual"]) for r in matured]
    pcts = [abs(r["predicted"] - r["actual"]) / r["actual"] * 100 for r in matured if r["actual"]]
    return {
        "scored_points": len(matured),
        "realised_MAE": round(sum(errs) / len(errs), 2),
        "realised_MAPE_pct": round(sum(pcts) / len(pcts), 2) if pcts else None,
        "recent": matured[-5:],
        "note": "Realised forecast accuracy, updated as forecast dates mature.",
    }


if __name__ == "__main__":
    import json as j
    log_forecast("BDI", ["2026-09-25", "2026-09-26"], [130.0, 131.0])
    print("logged. score:", j.dumps(score("BDI", [{"Date": "2026-09-25", "Close": 129.0}]), indent=2))
