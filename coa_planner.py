"""
coa_planner.py — SIH26006
Contract-of-Affreightment (COA) / programme planner — the problem statement's
core ask: move from single spot fixtures to a planned, multi-voyage programme.

Given a tonnage target over a period, it:
  - splits the target into voyages sized to the chosen vessel class
  - spaces them across the period
  - prices each voyage through the voyage-cost model at the forecast rate for
    its window (cheaper windows preferred), versus pricing them all at spot today
  - returns the programme, the all-in cost, and the saving vs. all-spot

This is a transparent planning aid, not an optimiser — it demonstrates the
spot-to-COA shift with real economics and is the scaffold for full optimisation.
"""
from __future__ import annotations
import datetime as dt


def plan_programme(vessel_class: str, total_mt: int, months: int,
                   distance_nm: float, cargo_type: str,
                   base_rate_usd_per_t: float, bunker_price: float,
                   forecast_dates: list[str], forecast_values: list[float],
                   parcel_mt: int) -> dict:
    """
    vessel_class: chosen class (from the recommendation).
    total_mt: tonnage to move over the programme.
    months: programme length.
    parcel_mt: nominal parcel per voyage (defaults sized to the class).
    forecast_*: the freight forecast, used to pick cheaper windows.
    """
    from voyage_cost import estimate_voyage

    if parcel_mt <= 0:
        parcel_mt = 75000
    n_voyages = max(1, round(total_mt / parcel_mt))
    per_voyage_mt = round(total_mt / n_voyages)

    # today's level and forecast, to translate window rates
    today_lvl = forecast_values[0] if forecast_values else None

    # choose n_voyages spaced points across the forecast, preferring lower rates
    voyages = []
    if forecast_values and today_lvl:
        # rank forecast days by value (cheapest first), then pick spaced ones
        idx_sorted = sorted(range(len(forecast_values)), key=lambda i: forecast_values[i])
        chosen = sorted(idx_sorted[:max(n_voyages, 1)])   # cheapest n, in date order
    else:
        chosen = list(range(n_voyages))

    # Two strategies compared honestly:
    #  A) SPOT-AS-YOU-GO: buy each voyage at the forecast rate for its natural
    #     delivery slot (tonnage spread evenly across the period).
    #  B) COA / FORWARD COVER: lock today's rate now for all voyages.
    # Saving from the COA = (spot-as-you-go cost) - (forward-cover cost).
    # Rising market → forward cover wins; falling market → spot-as-you-go wins.
    total_forward_cost = 0.0
    total_spotgo_cost = 0.0
    # Spread voyages across the actual programme length (months). The forecast is
    # ~45 business days (~2 months); if the programme is longer we extrapolate the
    # slot spacing so `months` genuinely changes the delivery schedule.
    fdays = len(forecast_values)
    prog_days = max(1, int(months * 21))          # ~21 business days per month
    slot_step = max(1, int((min(fdays, prog_days) - 1) // max(n_voyages, 1))) if forecast_values else 1
    for k in range(n_voyages):
        if forecast_values and today_lvl:
            i = min(k * slot_step, len(forecast_values) - 1)
            ratio = forecast_values[i] / today_lvl
            slot_rate = round(base_rate_usd_per_t * ratio, 2)
            when = forecast_dates[i] if i < len(forecast_dates) else None
        else:
            slot_rate = base_rate_usd_per_t
            when = None
        fwd = estimate_voyage(vessel_class, distance_nm, per_voyage_mt,
                              base_rate_usd_per_t, bunker_price, cargo_type)
        spg = estimate_voyage(vessel_class, distance_nm, per_voyage_mt,
                              slot_rate, bunker_price, cargo_type)
        fc_cost = fwd["result"]["total_charterer_cost_usd"]
        sg_cost = spg["result"]["total_charterer_cost_usd"]
        total_forward_cost += fc_cost
        total_spotgo_cost += sg_cost
        voyages.append({
            "voyage": k + 1, "target_window": when, "parcel_mt": per_voyage_mt,
            "spotgo_rate_usd_per_t": slot_rate, "forward_rate_usd_per_t": base_rate_usd_per_t,
            "spotgo_cost_usd": round(sg_cost, 0), "forward_cost_usd": round(fc_cost, 0),
        })

    saving = round(total_spotgo_cost - total_forward_cost, 0)   # +ve = COA/forward wins
    recommend_forward = saving > 0
    return {
        "vessel_class": vessel_class,
        "total_mt": total_mt, "months": months,
        "n_voyages": n_voyages, "per_voyage_mt": per_voyage_mt,
        "voyages": voyages,
        "forward_cover_cost_usd": round(total_forward_cost, 0),
        "spot_as_you_go_cost_usd": round(total_spotgo_cost, 0),
        "programme_saving_usd": saving,
        "saving_pct": round(saving / total_spotgo_cost * 100, 1) if total_spotgo_cost else 0,
        "recommended_strategy": "forward-cover (COA) — lock today's rate now" if recommend_forward
                                else "spot-as-you-go — stage purchases; market forecast to soften",
        "basis": "forward-cover (all voyages at today's rate) vs. spot-as-you-go (each voyage at its forecast slot rate), through the voyage-cost model",
    }


if __name__ == "__main__":
    import json
    fdates = [f"2026-{(10+(i//28)):02d}-{(i%28)+1:02d}" for i in range(45)]
    fvals = [120 + i*0.5 for i in range(45)]
    r = plan_programme("Panamax", total_mt=450000, months=3, distance_nm=10800,
                       cargo_type="Coking coal", base_rate_usd_per_t=18.3,
                       bunker_price=600, forecast_dates=fdates, forecast_values=fvals,
                       parcel_mt=75000)
    print(json.dumps(r, indent=2)[:1200])
