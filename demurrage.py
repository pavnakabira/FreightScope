"""
demurrage.py — SIH26006
Converts the live port-congestion signal into an expected-waiting-days figure
and a demurrage-cost risk, so congestion becomes money, not just a label.

Demurrage is the penalty a charterer pays when a vessel waits beyond agreed
laytime — often the single largest avoidable cost at busy Indian bulk ports.
Method (transparent, adjustable):
  expected_wait_days = base_wait * congestion_index    (queue-scaled)
  demurrage_cost     = max(0, expected_wait_days - free_days) * demurrage_rate
Rates are representative per class (USD/day) and shown as assumptions.
"""
from __future__ import annotations

# representative demurrage rates (USD/day) by class — order-of-magnitude, adjustable
DEMURRAGE_RATE = {
    "Handysize": 12000, "Supramax": 15000, "Panamax": 18000, "Capesize": 28000,
}
BASE_WAIT_DAYS = 2.0    # nominal wait at congestion_index = 1.0
FREE_DAYS = 1.0         # laytime buffer before demurrage accrues


def demurrage_estimate(vessel_class: str, congestion_index: float | None,
                       free_days: float = FREE_DAYS) -> dict:
    rate = DEMURRAGE_RATE.get(vessel_class, 15000)
    if congestion_index is None:
        return {"available": False,
                "note": "No live congestion index for this port; demurrage risk not estimated."}
    exp_wait = round(BASE_WAIT_DAYS * float(congestion_index), 1)
    billable = max(0.0, exp_wait - free_days)
    cost = round(billable * rate, 0)
    level = "high" if exp_wait > 2.5 else "moderate" if exp_wait > 1.5 else "low"
    return {
        "available": True,
        "vessel_class": vessel_class,
        "congestion_index": round(float(congestion_index), 3),
        "expected_wait_days": exp_wait,
        "free_days": free_days,
        "billable_days": round(billable, 1),
        "demurrage_rate_usd_per_day": rate,
        "demurrage_cost_usd": cost,
        "risk_level": level,
        "basis": "expected wait = base_wait × congestion_index; cost = billable days × rate",
    }


if __name__ == "__main__":
    import json
    for ci in (0.6, 0.9, 1.6):
        print(f"congestion {ci}:", json.dumps(demurrage_estimate("Panamax", ci)))
