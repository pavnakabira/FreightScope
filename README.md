# FreightScope Backend — SIH26006

FastAPI service that serves **real** freight forecasts and vessel-port feasibility to the dashboard.

## What it does
- Fetches **live** freight-driver data from **FRED** (Brent/WTI crude — free, citable, no key needed for the CSV endpoint).
- Loads **real historical Baltic index** data from CSVs you drop in `./data/` (for training/backtesting).
- Runs a **Prophet** time-series forecast with an 80% confidence band.
- Applies the **verified port + vessel feasibility** logic (same constraints as the workbook).
- Computes the **best entry window** (lowest forecast day in the horizon) and a plain-English verdict.

## Data sources — the honest picture
The true **Baltic Dry Index is licensed** by the Baltic Exchange; there is no free live API for the index itself. So the data layer uses real *proxies* that are free and live, in three tiers:
1. **Live, free, citable:**
   - **BDRY ETF** (Breakwave Dry Bulk Shipping ETF) via Yahoo Finance — holds Baltic freight futures, so it **tracks the BDI closely**. Used as the live Baltic signal. `BDI` maps to it directly; `BCI/BPI/BSI/BHSI` are *scaled approximations* of it (the ETF doesn't split out sub-indices — the source label says "scaled to X" to flag this).
   - **Brent/WTI crude** via FRED — a real freight-cost driver, used as a further fallback.
2. **Historical CSV** (`./data/`) — if you have real per-sub-index Baltic history, drop it in and it overrides the ETF approximation for that code.
3. **Synthetic** — deterministic, clearly labelled, so nothing breaks offline.

**Honest framing for the demo:** *"`BDI` is live and real (BDRY ETF tracks the Baltic Dry Index). Sub-indices are approximated from it, since true per-class Baltic data is licensed — in production we'd use SAIL's chartering feed or a Baltic Exchange subscription."* Every response's `source` field states exactly which tier produced it, so you're never overclaiming.

**One-time SSL note (Windows):** if `requests`/`yfinance` fail with `CERTIFICATE_VERIFY_FAILED`, run `pip install --upgrade certifi pip-system-certs` and open a fresh terminal. (yfinance may still print a harmless "Cookie fetch failed" line — it fetches the data regardless.)

## Setup (one time)
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
> Prophet pulls in `cmdstanpy`; first install can take a few minutes. If it struggles on Windows, install via conda: `conda install -c conda-forge prophet`.

## Run
```bash
uvicorn app:app --reload --port 8000
```
Open **http://localhost:8000/docs** for the interactive API (Swagger UI) — you can click every endpoint and see live JSON.

## Endpoints
| Endpoint | Example | Returns |
|---|---|---|
| `/health` | | service check |
| `/indices` | | latest BDI/BCI/BPI/BSI/BHSI + which source was used |
| `/forecast` | `/forecast?index=BPI&days=30` | history + forecast + confidence band + entry window |
| `/compare` | `/compare?index=BPI&horizon=7&folds=6` | Naive vs Prophet vs XGBoost backtest — MAE/RMSE/MAPE + skill vs naive |
| `/congestion` | `/congestion?days=14` | real Paradip berth-queue congestion from Daily Traffic Reports |
| `/recommend` | `/recommend?origin=Australia — Hay Point&dest=Paradip&cargo=Coking coal` | full recommendation: feasible vessels, pick, forecast, verdict |
| `/ports` | | verified port constraint table |

## Important notes for the demo
- **The forecast is real** when Stooq is reachable. If it isn't (some networks/firewalls block it), the service automatically falls back to a **deterministic synthetic series** and *says so* in the `source` field — it never crashes. Check the `source` string to know which you're showing.
- **Sub-index scaling is a proxy.** Stooq reliably serves the BDI composite; we scale it to BCI/BPI/etc. with fixed ratios. For a stronger submission, replace `SUBINDEX_TO_BALTIC` with real per-index data if you can source it (e.g. a paid Baltic feed, or scraped daily values from HandyBulk).
- **Prophet warning about history length** is expected and harmless — we feed ~180 days deliberately for speed.

## Wiring the dashboard to this backend
The dashboard currently generates illustrative numbers in-browser. To make it call this API instead, replace its `buildForecast()` and the recommend logic with `fetch()` calls, e.g.:
```js
const r = await fetch(`http://localhost:8000/recommend?origin=${encodeURIComponent(origin)}&dest=${dest}&cargo=${cargo}`);
const data = await r.json();
// data.recommended_class, data.vessels, data.forecast.entry_window, data.verdict
```
Run the backend first (port 8000), then open the dashboard from a local server (not file://) so the browser allows the request. CORS is already enabled on the backend.

## Model comparison 
The `/compare` endpoint runs a **walk-forward backtest** (no lookahead) scoring three models on the same held-out days:
- **Naive** (tomorrow = today) — the baseline every model must beat.
- **Prophet** — good for trend + seasonality; gives confidence bands.
- **XGBoost** — lag/rolling/calendar features; usually best on short daily horizons.



## Next steps to strengthen it
1. Adding the **Paradip Daily Traffic Report scraper** as a congestion feature (real berth-queue signal).
2. Adding **commodity price** (World Bank Pink Sheet) as an exogenous regressor in Prophet.
3. Swapping the cost function's flat `usd_per_t` for a real freight + idle + deadhead calc.
4. Adding an **XGBoost** model alongside Prophet and show the baseline-vs-advanced comparison.
