"""
Descarga datos REALES de Yahoo Finance y los transforma al formato exacto del hackathon:
  - train_indices.csv       — cierres ajustados diarios (días hábiles)
  - test_dates.csv          — fechas del año de test
  - train_macro_factors.csv — Oro, Crudo, Tipo de interés
  - test_macro_factors.csv
  - train_network_metrics.csv — métricas on-chain de Bitcoin (aprox. desde 2010)
  - test_network_metrics.csv
  - train_news.csv           — titulares reales vía yfinance .news (limitado)
  - test_news.csv

Mapeo de índices:
  Index_A  NASDAQ-100         → ^NDX
  Index_B  S&P 500            → ^GSPC
  Index_C  Bloomberg Commodity → DBC (ETF) o GSCI
  Index_D  RECONSTRUIDO: Index_A desplazado 3 días + ruido pequeño
  Index_E  MSCI World ESG     → ESGW (iShares) o URTH como proxy
  Index_F  Bitcoin            → BTC-USD

Macro:
  Gold         → GC=F  (Gold Futures)
  Oil          → CL=F  (Crude Oil WTI Futures)
  InterestRate → FEDFUNDS via FRED (o proxy ^IRX — T-Bill 13 semanas)

Requisitos:
  pip install yfinance pandas numpy requests

Uso:
  python download_real_data.py [--test-year 2024]
"""

import os
import argparse
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Ejecuta: pip install yfinance")

SEED = 42
np.random.seed(SEED)

OUT = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT, exist_ok=True)

# ── Mapeo de tickers ──────────────────────────────────────────────────────────
TICKERS = {
    "Index_A": "^NDX",       # NASDAQ-100
    "Index_B": "^GSPC",      # S&P 500
    "Index_C": "DBC",        # Bloomberg Commodity ETF (proxy)
    "Index_E": "URTH",       # iShares MSCI World ETF (proxy ESG global)
    "Index_F": "BTC-USD",    # Bitcoin
}
MACRO_TICKERS = {
    "Gold":         "GC=F",
    "Oil":          "CL=F",   # WTI Crude
    "InterestRate": "^IRX",   # T-Bill 13 semanas (proxy tipo libre de riesgo)
}


def download(ticker: str, start: str, end: str) -> pd.Series:
    """Descarga 'Close' ajustado y devuelve una Serie con índice de fechas."""
    print(f"  Descargando {ticker} ...", end=" ", flush=True)
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No se obtuvieron datos para {ticker}")
    # yfinance puede devolver MultiIndex en columnas si multi_level_column=True
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    print(f"OK  ({len(s)} filas, {s.index[0].date()} - {s.index[-1].date()})")
    return s


def align_to_bdays(series: pd.Series, bday_index: pd.DatetimeIndex) -> pd.Series:
    """
    Reindexar al calendario de días hábiles: forward-fill los huecos
    (fines de semana, festivos propios del activo, etc.).
    """
    return series.reindex(bday_index).ffill().bfill()


def make_news(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Genera titulares sintéticos — yfinance .news no tiene histórico largo."""
    templates = [
        "Tech stocks surge as {idx} breaks record high",
        "Analysts forecast growth in {idx} following earnings beat",
        "Energy prices impact {idx} amid geopolitical tensions",
        "{idx} volatility spikes on Federal Reserve announcement",
        "Sustainable investments boost {idx} performance",
        "Crypto market rally lifts {idx} to new levels",
        "{idx} declines as recession fears mount",
        "Global ESG trend supports {idx} momentum",
        "Oil price drop weighs on {idx} sector",
        "Central bank policy shift affects {idx} outlook",
    ]
    rows = []
    for d in dates:
        if np.random.rand() < 0.4:
            idx_name = np.random.choice(
                ["Index_A", "Index_B", "Index_C", "Index_D", "Index_E", "Index_F", "markets"]
            )
            tmpl = np.random.choice(templates)
            rows.append({"Date": d, "Headline": tmpl.format(idx=idx_name)})
    return pd.DataFrame(rows).set_index("Date")


def save(df: pd.DataFrame, name: str):
    path = os.path.join(OUT, name)
    df.to_csv(path)
    print(f"  {name:45s}  shape={str(df.shape):>15}  -> {path}")


def main(test_year: int = 2024):
    # ── Rango de fechas ───────────────────────────────────────────────────────
    # Test: el año completo indicado
    test_start = f"{test_year}-01-01"
    test_end   = f"{test_year}-12-31"

    # Train: todo lo disponible antes del año de test
    # Para NDX/GSPC hay datos desde 1985-1990, BTC desde 2014-09
    # Usamos 2015-01-01 como mínimo común para tener BTC
    train_start = "2015-01-01"
    train_end   = f"{test_year - 1}-12-31"

    FULL_START = train_start
    FULL_END   = test_end

    print(f"\nRango train: {train_start} - {train_end}")
    print(f"Rango test : {test_start} - {test_end}\n")

    # ── Calendario de días hábiles (bdays NYSE approx) ────────────────────────
    all_bdays = pd.bdate_range(start=FULL_START, end=FULL_END)
    train_bdays = all_bdays[all_bdays < test_start]
    test_bdays  = all_bdays[all_bdays >= test_start]

    print(f"Días hábiles train: {len(train_bdays)},  test: {len(test_bdays)}\n")

    # ── Descargar índices ─────────────────────────────────────────────────────
    print("=== Índices ===")
    raw = {}
    for col, ticker in TICKERS.items():
        raw[col] = align_to_bdays(download(ticker, FULL_START, FULL_END), all_bdays)

    # Index_D: Index_A desplazado 3 días hábiles + ruido 0.5%
    lag = 3
    ghost_noise = 0.005
    A_arr = raw["Index_A"].values
    D_arr = np.concatenate([np.full(lag, A_arr[0]), A_arr[:-lag]])
    D_arr = D_arr * (1 + ghost_noise * np.random.randn(len(D_arr)))
    raw["Index_D"] = pd.Series(D_arr, index=all_bdays)

    idx_full = pd.DataFrame({
        "Index_A": raw["Index_A"],
        "Index_B": raw["Index_B"],
        "Index_C": raw["Index_C"],
        "Index_D": raw["Index_D"],
        "Index_E": raw["Index_E"],
        "Index_F": raw["Index_F"],
    }, index=all_bdays)
    idx_full.index.name = "Date"

    # ── Descargar macro ───────────────────────────────────────────────────────
    print("\n=== Macro factors ===")
    gold_s  = align_to_bdays(download(MACRO_TICKERS["Gold"],         FULL_START, FULL_END), all_bdays)
    oil_s   = align_to_bdays(download(MACRO_TICKERS["Oil"],          FULL_START, FULL_END), all_bdays)
    rate_s  = align_to_bdays(download(MACRO_TICKERS["InterestRate"], FULL_START, FULL_END), all_bdays)

    # ^IRX cotiza en % anualizado, dividir entre 100 para tener fracción
    rate_s = rate_s / 100.0

    macro_full = pd.DataFrame({
        "Gold":         gold_s,
        "Oil":          oil_s,
        "InterestRate": rate_s,
    }, index=all_bdays)
    macro_full.index.name = "Date"

    # ── Network metrics (Bitcoin on-chain, proxy con derivados del precio) ────
    # No hay API pública gratuita de on-chain histórico; usamos proxies estadísticos
    # basados en BTC-USD:
    #   ActiveNodes  ~ abs(rolling std de volumen × escala)
    #   TxVolume     ~ volumen de BTC (shares traded) × precio
    #   HashRate     ~ índice de dificultad (proxy: rolling 30d ret acumulado)
    print("\n=== Network metrics (proxy Bitcoin) ===")
    btc_raw = yf.download("BTC-USD", start=FULL_START, end=FULL_END,
                          auto_adjust=True, progress=False)
    if isinstance(btc_raw.columns, pd.MultiIndex):
        btc_raw.columns = btc_raw.columns.get_level_values(0)
    btc_raw.index = pd.to_datetime(btc_raw.index).tz_localize(None)
    btc_close  = btc_raw["Close"].reindex(all_bdays).ffill().bfill()
    btc_volume = btc_raw["Volume"].reindex(all_bdays).ffill().bfill().fillna(0)

    active_nodes = (btc_close.rolling(30).std().bfill() * 100).clip(lower=1000)
    tx_volume    = (btc_close * btc_volume).clip(lower=1)
    hash_rate    = (btc_close.pct_change().rolling(30).mean().fillna(0) * 1000 + 200).clip(lower=10)

    net_full = pd.DataFrame({
        "ActiveNodes": active_nodes,
        "TxVolume":    tx_volume,
        "HashRate":    hash_rate,
    }, index=all_bdays)
    net_full.index.name = "Date"

    # ── Noticias (sintéticas — no hay histórico libre largo) ─────────────────
    print("\n=== Noticias (sintéticas) ===")
    train_news = make_news(train_bdays)
    test_news  = make_news(test_bdays)

    # ── Guardar ───────────────────────────────────────────────────────────────
    print("\n=== Guardando CSVs ===")
    save(idx_full.loc[train_bdays],   "train_indices.csv")
    save(pd.DataFrame(index=test_bdays).rename_axis("Date"), "test_dates.csv")
    save(macro_full.loc[train_bdays], "train_macro_factors.csv")
    save(macro_full.loc[test_bdays],  "test_macro_factors.csv")
    save(net_full.loc[train_bdays],   "train_network_metrics.csv")
    save(net_full.loc[test_bdays],    "test_network_metrics.csv")
    save(train_news,                  "train_news.csv")
    save(test_news,                   "test_news.csv")

    # ── Verificación ──────────────────────────────────────────────────────────
    train_idx_df = idx_full.loc[train_bdays]
    corr_D_A = train_idx_df["Index_D"].corr(train_idx_df["Index_A"].shift(3).dropna())
    print(f"\nVerificacion: correlacion Index_D vs Index_A lag-3 = {corr_D_A:.4f}")
    print(f"NaN en train_indices: {train_idx_df.isna().sum().sum()}")
    print(f"NaN en train_macro:   {macro_full.loc[train_bdays].isna().sum().sum()}")
    print("\nListo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga datos reales para el hackathon")
    parser.add_argument(
        "--test-year", type=int, default=2024,
        help="Año de test (el año completo que queda fuera de train). Default: 2024"
    )
    args = parser.parse_args()
    main(test_year=args.test_year)
