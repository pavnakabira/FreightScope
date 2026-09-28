"""
proxy_quality.py — SIH26006
Quantifies how well the BDRY ETF proxies the Baltic Dry Index, turning a
disclosed assumption into a defended number.

Method: align BDRY and a BDI reference series on common dates, compute
  - Pearson correlation of levels
  - Pearson correlation of daily returns (the stricter, more honest test)
  - tracking error (std of return differences, annualised)
over the overlapping window.

BDI reference: if a real BDI history CSV is present in ./data/bdi_history.csv it
is used; otherwise the function reports that no reference is available (it will
NOT fabricate a correlation). On the user's machine, dropping a BDI CSV in place
yields a real, citable figure.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _returns(s: pd.Series) -> pd.Series:
    return s.pct_change().dropna()


def proxy_quality(bdry_df: pd.DataFrame, bdi_df: pd.DataFrame | None) -> dict:
    """
    bdry_df, bdi_df: DataFrames with columns Date, Close.
    Returns correlation + tracking-error metrics, or a clear 'no reference' note.
    """
    if bdi_df is None or bdi_df.empty:
        return {"available": False,
                "note": "No BDI reference series found. Drop data/bdi_history.csv "
                        "to compute a real BDRY-BDI correlation."}
    a = bdry_df[["Date", "Close"]].rename(columns={"Close": "bdry"})
    b = bdi_df[["Date", "Close"]].rename(columns={"Close": "bdi"})
    m = pd.merge(a, b, on="Date", how="inner").sort_values("Date")
    if len(m) < 30:
        return {"available": False,
                "note": f"Only {len(m)} overlapping dates — too few for a reliable figure."}
    lvl_corr = float(np.corrcoef(m["bdry"], m["bdi"])[0, 1])
    ra, rb = _returns(m["bdry"]), _returns(m["bdi"])
    n = min(len(ra), len(rb))
    ret_corr = float(np.corrcoef(ra.iloc[-n:], rb.iloc[-n:])[0, 1])
    diff = (ra.iloc[-n:].values - rb.iloc[-n:].values)
    te_annual = float(np.std(diff) * np.sqrt(252))
    return {
        "available": True,
        "overlap_days": int(len(m)),
        "level_correlation": round(lvl_corr, 3),
        "return_correlation": round(ret_corr, 3),
        "tracking_error_annualised": round(te_annual, 3),
        "window": {"from": m["Date"].iloc[0].strftime("%Y-%m-%d"),
                   "to": m["Date"].iloc[-1].strftime("%Y-%m-%d")},
        "interpretation": _interpret(lvl_corr, ret_corr),
    }


def _interpret(lvl, ret):
    if lvl >= 0.9:
        base = "Strong level tracking — BDRY closely follows the BDI's trajectory."
    elif lvl >= 0.75:
        base = "Good level tracking — BDRY follows the BDI's broad movements."
    else:
        base = "Moderate level tracking — use BDRY as a directional proxy only."
    if ret is not None and ret >= 0.6:
        base += " Day-to-day returns are also well aligned."
    elif ret is not None and ret >= 0.3:
        base += " Short-term returns diverge somewhat, as expected for an ETF vs. the index."
    else:
        base += " Short-term returns diverge; treat as a trend proxy, not a tick-level match."
    return base


def load_bdi_reference() -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, "bdi_history.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        dcol = next((cols[c] for c in cols if "date" in c or "observation" in c), df.columns[0])
        vcol = next((cols[c] for c in cols if any(k in c for k in ("close","value","price","bdi","index"))), df.columns[-1])
        out = df[[dcol, vcol]].copy(); out.columns = ["Date", "Close"]
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
        return out.dropna().sort_values("Date").reset_index(drop=True)
    except Exception:
        return None


if __name__ == "__main__":
    # synthetic self-test: build correlated series and confirm the math
    rng = np.random.default_rng(1)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=300)
    bdi = 1500 + np.cumsum(rng.normal(0, 20, 300))
    bdry = bdi * 0.083 + rng.normal(0, 1.5, 300)   # scaled + noise
    A = pd.DataFrame({"Date": dates, "Close": bdry})
    B = pd.DataFrame({"Date": dates, "Close": bdi})
    import json
    print(json.dumps(proxy_quality(A, B), indent=2))
