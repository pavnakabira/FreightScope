"""
idle_repositioning.py — SIH26006
Deliverable (c): idle-scenario management & deadhead economics.

Turns the forecast + fleet-position picture into concrete repositioning guidance:
  - detect forecast demand troughs (low-rate windows = idle risk)
  - estimate ballast (deadhead) cost of repositioning an empty vessel
  - recommend hold vs. reposition vs. take-a-lower-rate-fixture

Ballast cost uses the same voyage-economics basis as voyage_cost.py:
  ballast_days  = distance_nm / (speed*24) * (1+weather)
  ballast_cost  = ballast_days * (sea_cons * bunker_price + daily_opex)
A vessel idle at anchor still burns OPEX; repositioning adds bunkers. The tool
compares the cost of waiting vs. moving to where the next cargo likely is.
"""
from __future__ import annotations
import numpy as np

# reuse representative params (kept local to avoid tight coupling)
SHIP = {
    "Handysize": {"speed": 13.0, "sea_cons": 18.0, "opex": 5500},
    "Supramax":  {"speed": 13.5, "sea_cons": 22.0, "opex": 6000},
    "Panamax":   {"speed": 13.5, "sea_cons": 26.0, "opex": 6500},
    "Capesize":  {"speed": 13.0, "sea_cons": 40.0, "opex": 8000},
}
WEATHER = 0.05


def detect_troughs(fc_values: list[float], fc_dates: list[str], pct_below: float = 0.05) -> list[dict]:
    """Flag forecast days that are genuine soft-demand dips: materially below the
    horizon mean AND below the local trend (a real dip, not just the low side of
    a smooth rise). This avoids over-flagging on a steadily trending forecast."""
    if not fc_values or len(fc_values) < 5:
        return []
    arr = np.array(fc_values, dtype=float)
    mean = arr.mean()
    # local baseline = centered rolling mean (window 5)
    n = len(arr)
    troughs = []
    for i in range(n):
        lo = max(0, i - 2); hi = min(n, i + 3)
        local = arr[lo:hi].mean()
        below_mean = arr[i] < mean * (1 - pct_below)
        below_local = arr[i] < local * (1 - pct_below * 0.6)   # a genuine local dip
        if below_mean and below_local:
            troughs.append({"date": fc_dates[i] if i < len(fc_dates) else None,
                            "value": round(float(arr[i]), 1),
                            "pct_below_mean": round((mean - arr[i]) / mean * 100, 1)})
    return troughs


def ballast_cost(vessel_class: str, distance_nm: float, bunker_price: float = 600.0) -> dict:
    """Cost to reposition an empty vessel `distance_nm` to the next load area."""
    s = SHIP.get(vessel_class, SHIP["Panamax"])
    days = distance_nm / (s["speed"] * 24.0) * (1 + WEATHER)
    bunkers = days * s["sea_cons"] * bunker_price
    opex = days * s["opex"]
    total = bunkers + opex
    return {"vessel_class": vessel_class, "ballast_distance_nm": round(distance_nm, 0),
            "ballast_days": round(days, 1), "bunker_cost_usd": round(bunkers, 0),
            "opex_cost_usd": round(opex, 0), "total_ballast_cost_usd": round(total, 0)}


def idle_analysis(vessel_class: str, fc_values: list[float], fc_dates: list[str],
                  reposition_nm: float = 0.0, bunker_price: float = 600.0) -> dict:
    """
    Full idle/repositioning view for the recommended class.
    reposition_nm: distance to the nearest alternative load region (0 = unknown/skip).
    """
    s = SHIP.get(vessel_class, SHIP["Panamax"])
    troughs = detect_troughs(fc_values, fc_dates)
    out = {"vessel_class": vessel_class,
           "trough_count": len(troughs),
           "troughs": troughs[:5],
           "idle_opex_per_day_usd": s["opex"]}
    if reposition_nm and reposition_nm > 0:
        out["reposition"] = ballast_cost(vessel_class, reposition_nm, bunker_price)
    # simple guidance
    if not troughs:
        out["guidance"] = "No material demand trough in the horizon — standard rotation viable; no repositioning needed."
    elif len(troughs) <= 2:
        out["guidance"] = (f"Short soft-demand window ({len(troughs)} day(s)). Likely cheaper to wait "
                           f"at ~${s['opex']:,}/day OPEX than to ballast away; hold position.")
    else:
        rc = out.get("reposition", {}).get("total_ballast_cost_usd")
        idle_cost = len(troughs) * s["opex"]
        if rc and rc < idle_cost:
            out["guidance"] = (f"Extended soft demand ({len(troughs)} days ≈ ${idle_cost:,} idle OPEX). "
                               f"Repositioning (~${rc:,}) is cheaper — consider ballasting to the alternative load region.")
        else:
            out["guidance"] = (f"Extended soft demand ({len(troughs)} days). Seek backhaul/short coastal "
                               f"employment to cover idle OPEX (~${idle_cost:,}) rather than ballasting empty.")
    return out


if __name__ == "__main__":
    import json
    vals = [120,119,118,121,122,115,114,116,123,124,125,126]
    dates = [f"2026-10-{d:02d}" for d in range(1,13)]
    print(json.dumps(idle_analysis("Panamax", vals, dates, reposition_nm=1800), indent=2))
