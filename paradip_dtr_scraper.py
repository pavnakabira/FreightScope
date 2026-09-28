"""
Paradip Daily Traffic Report (DTR) scraper — SIH26006
Extracts REAL port-congestion signals from Paradip Port's daily PDFs.

What the DTRs actually contain (verified against live PDFs, Sep 2026):
  - Vessels WORKING (berthed & operating)
  - Vessels WAITING AT ANCHORAGE   <- queue length = congestion signal
  - EXPECTED vessels
  - Daily cargo throughput (MT) and vessels berthed/sailed counts
They do NOT reliably give per-vessel LOA/draft/DWT, so we extract the
congestion metrics, which is what the risk-mitigation module needs.

URL format (current): https://www.paradipport.gov.in/uploads/YYYY/MM/dtrDDMM.pdf
  (case varies: dtr / DTR). Older archive: /Writereaddata/Daily_Traffic/dtrDDMM.pdf

Usage:
    python paradip_dtr_scraper.py --days 30 --out paradip_congestion.csv
    # or import build_congestion_series() from another module
"""
from __future__ import annotations
import argparse, io, re, sys, datetime as dt
from typing import Optional
import requests

try:
    import pdfplumber
except ImportError:
    sys.exit("pip install pdfplumber requests")

HEADERS = {"User-Agent": "Mozilla/5.0 (SIH26006 research scraper)"}

def candidate_urls(d: dt.date) -> list[str]:
    dd, mm, yyyy = d.strftime("%d"), d.strftime("%m"), d.strftime("%Y")
    stem = f"dtr{dd}{mm}.pdf"
    return [
        f"https://www.paradipport.gov.in/uploads/{yyyy}/{mm}/{stem}",
        f"https://www.paradipport.gov.in/uploads/{yyyy}/{mm}/DTR{dd}{mm}.pdf",
        f"https://paradipport.gov.in/Writereaddata/Daily_Traffic/{stem}",
    ]

def fetch_pdf_text(d: dt.date) -> Optional[str]:
    for url in candidate_urls(d):
        try:
            r = requests.get(url, headers=HEADERS, timeout=8)
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                with pdfplumber.open(io.BytesIO(r.content)) as pdf:
                    return "\n".join((p.extract_text() or "") for p in pdf.pages)
        except Exception:
            continue
    return None

# ---- parsing helpers -------------------------------------------------------
VESSEL_RE = re.compile(r"\bM[VT]\.\s+[A-Z0-9]")  # "MV. NAME" / "MT. NAME"

def _section(text: str, start_markers: list[str], end_markers: list[str]) -> str:
    up = text.upper()
    s = -1
    for m in start_markers:
        s = up.find(m)
        if s != -1:
            break
    if s == -1:
        return ""
    e = len(text)
    for m in end_markers:
        idx = up.find(m, s + 1)
        if idx != -1:
            e = min(e, idx)
    return text[s:e]

def count_vessels(segment: str) -> int:
    return len(VESSEL_RE.findall(segment))

def parse_dtr(text: str, d: dt.date) -> dict:
    working  = _section(text, ["A. WORKING VESSELS", "WORKING VESSELS AS ON"],
                               ["B. VESSELS WAITING", "VESSELS WAITING AT ANCHORAGE"])
    waiting  = _section(text, ["B. VESSELS WAITING", "VESSELS WAITING AT ANCHORAGE"],
                               ["C. EXPECTED", "EXPECTED VESSSEL", "EXPECTED VESSEL"])
    expected = _section(text, ["C. EXPECTED", "EXPECTED VESSSEL", "EXPECTED VESSEL"],
                               ["D. BERTHING", "MAXIMUM OF TWO COASTAL", "No. of Pilotage"])

    def find_int(pattern):
        m = re.search(pattern, text, re.IGNORECASE)
        return int(m.group(1)) if m else None

    berthed = find_int(r"DAYS VESSEL BERTHED\s*::\s*DT[:\d.]+\s*::\s*(\d+)")
    sailed  = find_int(r"DAYS VESSEL SAILOFF\s*::\s*DT[:\d.]+\s*::\s*(\d+)")
    cargo   = find_int(r"DAYS CARGO\s*::\s*DT[:\d.]+\s*::\s*(\d+)\s*MT")

    n_wait = count_vessels(waiting)
    n_work = count_vessels(working)
    n_exp  = count_vessels(expected)
    # congestion index: queue relative to berth activity (0..1+, higher = worse)
    denom = (n_work or berthed or 1)
    congestion = round(n_wait / denom, 3)

    return {
        "date": d.isoformat(),
        "vessels_working": n_work,
        "vessels_waiting": n_wait,
        "vessels_expected": n_exp,
        "vessels_berthed_today": berthed,
        "vessels_sailed_today": sailed,
        "cargo_mt_today": cargo,
        "congestion_index": congestion,
    }

# ---- public API ------------------------------------------------------------
def build_congestion_series(days: int = 30, end: Optional[dt.date] = None) -> list[dict]:
    """Walk back `days` calendar days from `end`, scraping each available DTR."""
    end = end or dt.date.today()
    rows, misses = [], 0
    for i in range(days):
        d = end - dt.timedelta(days=i)
        text = fetch_pdf_text(d)
        if not text:
            misses += 1
            continue
        try:
            rows.append(parse_dtr(text, d))
            print(f"  [{d}] working={rows[-1]['vessels_working']} "
                  f"waiting={rows[-1]['vessels_waiting']} "
                  f"congestion={rows[-1]['congestion_index']}", file=sys.stderr)
        except Exception as e:
            print(f"  [{d}] parse error: {e}", file=sys.stderr)
            misses += 1
    rows.sort(key=lambda r: r["date"])
    print(f"Scraped {len(rows)} reports, {misses} unavailable.", file=sys.stderr)
    return rows

def to_csv(rows: list[dict], path: str):
    import csv
    if not rows:
        print("No rows to write.", file=sys.stderr); return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {path}")

# ---- self-test on a synthetic DTR-shaped string (works offline) ------------
SAMPLE = """PARADIP PORT AUTHORITY TRAFFIC DEPARTMENT DAILY TRAFFIC UPDATE FOR 21-07-2026
DAYS CARGO :: DT:21.07.26 :: 433904 MT.| CUM FOR JUL.26 :: 7060222 MT.
DAYS VESSEL BERTHED :: DT:21.07.26 :: 6 | CUM FOR JUL.26 :: 159
DAYS VESSEL SAILOFF :: DT:21.07.26 :: 7 | CUM FOR JUL.26 :: 155
A. WORKING VESSELS AS ON
MV. ATLANTIC VISION MV. RIPLEY PROSPERITY MV. APJ SETHU MV. SEA SPIRIT
MV. ALAM SAYANG MV. KSL RUIYANG MV. GLORY V
B. VESSELS WAITING AT ANCHORAGE
MV. APJ INDRANI MV. CHOLA MELODY MT. SWARNA KALASH MT. AFRAPEARL II
MV. LUCILIA C MV. OCEAN AMITIE MV. ECO LEGACY MV. SPRING AMIR
MV. BRIGHT WIND MV. JABAL SAMHAN
C. EXPECTED VESSSEL
MV. FLAMINIA MV. FRATERNELLE MV. LC MILADY
MAXIMUM OF TWO COASTAL VESSELS WILL BE TAKEN AT A TIME
"""

def _selftest():
    row = parse_dtr(SAMPLE, dt.date(2026, 7, 21))
    print("SELF-TEST parsed:", row)
    assert row["vessels_working"] == 7, row
    assert row["vessels_waiting"] == 10, row
    assert row["vessels_expected"] == 3, row
    assert row["vessels_berthed_today"] == 6
    assert row["cargo_mt_today"] == 433904
    assert row["congestion_index"] == round(10/7, 3)
    print("SELF-TEST passed ✓")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--out", default="paradip_congestion.csv")
    ap.add_argument("--selftest", action="store_true", help="run offline parser test")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        rows = build_congestion_series(days=args.days)
        to_csv(rows, args.out)
