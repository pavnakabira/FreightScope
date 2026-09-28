"""
voyage_cost.py — SIH26006
Transparent dry-bulk voyage-cost model, replacing the placeholder $/tonne.

Method follows standard industry voyage estimation (Platts / HandyBulk / operator
practice), computing an itemised cost breakdown a chartering desk would recognise:

    sea_days      = distance_nm / (speed_kn * 24) * (1 + weather_allowance)
    port_days     = cargo/load_rate + cargo/discharge_rate
    bunker_cost   = (sea_days*sea_cons + port_days*port_cons) * bunker_price
    voyage_cost   = bunker_cost + port_DA(load) + port_DA(discharge) + canal_dues
    freight_cost  = freight_rate_usd_per_t * cargo            # what the buyer pays for freight
    total_cost    = freight_cost + voyage_cost                # landed freight cost to the charterer
    usd_per_tonne = total_cost / cargo
    TCE           = (freight_revenue - voyage_cost) / voyage_days   # owner-side sanity metric

Every figure is returned in the breakdown so nothing is a black box. Vessel-class
parameters are representative industry values (speed, consumption, port costs) and
are clearly documented as assumptions the user can override.
"""
from __future__ import annotations

# Representative technical & cost parameters per class (industry-typical, 2026).
# These are ASSUMPTIONS, exposed and overridable — not hidden magic numbers.
VESSEL_PARAMS = {
    "Handysize": {"speed_kn": 13.0, "sea_cons_t": 18.0, "port_cons_t": 3.0, "port_da_usd": 35000},
    "Supramax":  {"speed_kn": 13.5, "sea_cons_t": 22.0, "port_cons_t": 4.0, "port_da_usd": 40000},
    "Panamax":   {"speed_kn": 13.5, "sea_cons_t": 26.0, "port_cons_t": 5.0, "port_da_usd": 45000},
    "Capesize":  {"speed_kn": 13.0, "sea_cons_t": 40.0, "port_cons_t": 6.0, "port_da_usd": 60000},
}

# Cargo load/discharge rates (MT/day) — representative bulk terminal rates.
CARGO_RATES = {
    "Coking coal":  {"load": 40000, "discharge": 25000},
    "Thermal coal": {"load": 45000, "discharge": 30000},
    "Iron ore":     {"load": 60000, "discharge": 35000},
    "Limestone":    {"load": 30000, "discharge": 20000},
}
DEFAULT_CARGO_RATE = {"load": 40000, "discharge": 25000}

WEATHER_ALLOWANCE = 0.05      # +5% on sea time for weather/currents (standard margin)
BROKER_COMMISSION = 0.0125    # 1.25% typical
DAILY_OPEX = {                # owner daily operating cost, for TCE context
    "Handysize": 5500, "Supramax": 6000, "Panamax": 6500, "Capesize": 8000,
}


def estimate_voyage(
    vessel_class: str,
    distance_nm: float,
    cargo_mt: float,
    freight_rate_usd_per_t: float,
    bunker_price_usd_per_t: float = 600.0,
    cargo_type: str = "Coking coal",
    canal_dues_usd: float = 0.0,
) -> dict:
    """
    Returns a fully itemised voyage-cost breakdown.
    freight_rate_usd_per_t is the market freight rate (from the forecast layer).
    bunker_price_usd_per_t defaults to a representative VLSFO level; wire to a live
    bunker feed later. All intermediate values are returned for transparency.
    """
    vp = VESSEL_PARAMS.get(vessel_class)
    if vp is None or cargo_mt <= 0:
        return {"error": f"no parameters for {vessel_class} / invalid cargo"}

    rates = CARGO_RATES.get(cargo_type, DEFAULT_CARGO_RATE)

    # 1. time
    sea_days = distance_nm / (vp["speed_kn"] * 24.0) * (1.0 + WEATHER_ALLOWANCE)
    port_days = cargo_mt / rates["load"] + cargo_mt / rates["discharge"]
    voyage_days = sea_days + port_days

    # 2. bunkers
    bunker_tonnes = sea_days * vp["sea_cons_t"] + port_days * vp["port_cons_t"]
    bunker_cost = bunker_tonnes * bunker_price_usd_per_t

    # 3. port disbursements (both ends) + canal
    port_costs = 2 * vp["port_da_usd"]
    voyage_cost = bunker_cost + port_costs + canal_dues_usd   # owner voyage-related cost

    # 4. freight & landed cost to the charterer (buyer perspective)
    freight_cost = freight_rate_usd_per_t * cargo_mt
    commission = freight_cost * BROKER_COMMISSION
    total_charterer_cost = freight_cost + commission          # what the buyer pays to move the cargo
    usd_per_tonne = total_charterer_cost / cargo_mt

    # 5. TCE (owner-side sanity metric): net daily earning of the fixture
    freight_revenue_net = freight_cost - commission
    tce_per_day = (freight_revenue_net - voyage_cost) / voyage_days if voyage_days else 0.0

    return {
        "vessel_class": vessel_class,
        "inputs": {
            "distance_nm": round(distance_nm, 0),
            "cargo_mt": round(cargo_mt, 0),
            "freight_rate_usd_per_t": round(freight_rate_usd_per_t, 2),
            "bunker_price_usd_per_t": round(bunker_price_usd_per_t, 2),
            "cargo_type": cargo_type,
        },
        "assumptions": {
            "speed_kn": vp["speed_kn"], "sea_cons_t_per_day": vp["sea_cons_t"],
            "port_cons_t_per_day": vp["port_cons_t"], "port_da_usd_each_end": vp["port_da_usd"],
            "load_rate_mt_day": rates["load"], "discharge_rate_mt_day": rates["discharge"],
            "weather_allowance_pct": WEATHER_ALLOWANCE * 100, "commission_pct": BROKER_COMMISSION * 100,
        },
        "breakdown": {
            "sea_days": round(sea_days, 1),
            "port_days": round(port_days, 1),
            "voyage_days": round(voyage_days, 1),
            "bunker_tonnes": round(bunker_tonnes, 0),
            "bunker_cost_usd": round(bunker_cost, 0),
            "port_costs_usd": round(port_costs, 0),
            "canal_dues_usd": round(canal_dues_usd, 0),
            "freight_cost_usd": round(freight_cost, 0),
            "commission_usd": round(commission, 0),
        },
        "result": {
            "total_charterer_cost_usd": round(total_charterer_cost, 0),
            "usd_per_tonne": round(usd_per_tonne, 2),
            "owner_voyage_cost_usd": round(voyage_cost, 0),
            "tce_per_day_usd": round(tce_per_day, 0),
            "opex_per_day_usd": DAILY_OPEX.get(vessel_class),
        },
    }


if __name__ == "__main__":
    # sanity check against the search examples (Panamax, ~$18/t coal)
    r = estimate_voyage("Panamax", distance_nm=10800, cargo_mt=75000,
                        freight_rate_usd_per_t=18.3, bunker_price_usd_per_t=600,
                        cargo_type="Coking coal")
    import json
    print(json.dumps(r, indent=2))
