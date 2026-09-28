# ./data/ — historical freight index CSVs

Drop real historical Baltic index CSVs here to replace the synthetic fallback.
The loader auto-detects a date column and a value column, so most formats work.

Expected filenames (only add the ones you have):
  bdi_history.csv    Baltic Dry Index (composite)
  bci_history.csv    Baltic Capesize Index
  bpi_history.csv    Baltic Panamax Index
  bsi_history.csv    Baltic Supramax Index
  bhsi_history.csv   Baltic Handysize Index

Minimal format (a date column + a value column; header names are flexible):
  date,BDI
  2024-01-02,2094
  2024-01-03,2113
  ...

Where to get a free historical BDI CSV:
  - Kaggle: search "Baltic Dry Index" — several public datasets with daily history.
  - GitHub: search "baltic dry index csv" — community-maintained series.
  - investing.com / tradingeconomics: export the BDI chart to CSV (free account).
  - Your own: if you have Bloomberg/Refinitiv access at CHRIST, export daily BDI.

Note: the LIVE Baltic Dry Index is licensed by the Baltic Exchange and has no
free real-time API. These CSVs give real *historical* values for training and
backtesting the models. For a live-updating freight signal, the backend uses
Brent crude from FRED as a documented proxy, and in production you would swap in
SAIL's real chartering-rate feed or a licensed Baltic Exchange subscription.
