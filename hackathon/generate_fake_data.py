"""
Genera datos ficticios en data/ que replican la estructura exacta del hackathon:
  - train_indices.csv       (11956 x 6)  — cierres diarios
  - test_dates.csv          (252 x 1)    — fechas a predecir
  - train_macro_factors.csv (11956 x 3)  — Oro, Crudo, Tipos
  - test_macro_factors.csv  (252 x 3)
  - train_network_metrics.csv (11956 x 3) — métricas on-chain Index_F
  - test_network_metrics.csv  (252 x 3)
  - train_news.csv           (variable x 2) — fecha + titular
  - test_news.csv            (variable x 2)

Propiedades de los índices (según el PDF):
  Index_A  Alta volatilidad, tendencia creciente (tech)
  Index_B  Baja volatilidad, defensivo
  Index_C  Relacionado con Crudo/Oro (macro)
  Index_D  "Ghost": sigue a Index_A con lag=3 + ruido pequeño
  Index_E  ESG / global, correlación moderada con Index_B
  Index_F  Volatilidad extrema, correlación con métricas on-chain
"""
import os
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

OUT = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT, exist_ok=True)

# ── Parámetros ────────────────────────────────────────────────
N_TRAIN  = 11956   # ~47 años de días hábiles
N_TEST   = 252     # 1 año
# Los valores están en un rango "desplazado" (disclaimer del PDF) → escala ~10k-500k
STARTS = {
    "Index_A": 15_000,
    "Index_B":  8_000,
    "Index_C": 25_000,
    "Index_D": 14_800,   # sigue a A
    "Index_E": 11_000,
    "Index_F":  3_000,
}

# ── Fechas ────────────────────────────────────────────────────
# Días hábiles (aprox.): generamos desde una fecha arbitraria
all_dates = pd.bdate_range(start="1975-01-02", periods=N_TRAIN + N_TEST)
train_dates = all_dates[:N_TRAIN]
test_dates  = all_dates[N_TRAIN:]

# ── Función de caminata aleatoria con GBM ─────────────────────
def gbm(start, mu, sigma, n):
    """Geometric Brownian Motion discreto."""
    dt = 1 / 252
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(n)
    prices = start * np.exp(np.cumsum(log_returns))
    return np.concatenate([[start], prices[:-1]])

# ── Generar precios ───────────────────────────────────────────
A_full = gbm(STARTS["Index_A"], mu=0.10, sigma=0.25, n=N_TRAIN + N_TEST)  # alta vol
B_full = gbm(STARTS["Index_B"], mu=0.04, sigma=0.08, n=N_TRAIN + N_TEST)  # baja vol
C_full = gbm(STARTS["Index_C"], mu=0.06, sigma=0.18, n=N_TRAIN + N_TEST)  # macro

# Index_D = Index_A desplazado 3 días + ruido pequeño (~5%)
lag = 3
ghost_noise = 0.005
D_base = np.concatenate([np.full(lag, STARTS["Index_D"]), A_full[:-lag]])
D_full = D_base * (1 + ghost_noise * np.random.randn(N_TRAIN + N_TEST))

E_full = gbm(STARTS["Index_E"], mu=0.05, sigma=0.12, n=N_TRAIN + N_TEST)  # ESG
F_full = gbm(STARTS["Index_F"], mu=0.15, sigma=0.60, n=N_TRAIN + N_TEST)  # crypto

train_idx = pd.DataFrame({
    "Index_A": A_full[:N_TRAIN],
    "Index_B": B_full[:N_TRAIN],
    "Index_C": C_full[:N_TRAIN],
    "Index_D": D_full[:N_TRAIN],
    "Index_E": E_full[:N_TRAIN],
    "Index_F": F_full[:N_TRAIN],
}, index=train_dates)
train_idx.index.name = "Date"

# ── Macro factors ─────────────────────────────────────────────
gold  = gbm(1800, mu=0.03, sigma=0.15, n=N_TRAIN + N_TEST)
oil   = gbm(80,   mu=0.02, sigma=0.30, n=N_TRAIN + N_TEST)
rates = np.clip(
    np.cumsum(0.002 * np.random.randn(N_TRAIN + N_TEST)) + 0.04, 0.001, 0.20)

macro_full = pd.DataFrame({
    "Gold":          gold,
    "Oil":           oil,
    "InterestRate":  rates,
}, index=all_dates)
macro_full.index.name = "Date"

# ── Network metrics (Index_F on-chain) ────────────────────────
active_nodes = np.abs(50_000 + np.cumsum(500 * np.random.randn(N_TRAIN + N_TEST)))
tx_volume    = np.abs(1e6 + np.cumsum(2e4 * np.random.randn(N_TRAIN + N_TEST)))
hash_rate    = np.abs(200 + np.cumsum(0.5 * np.random.randn(N_TRAIN + N_TEST)))

net_full = pd.DataFrame({
    "ActiveNodes": active_nodes,
    "TxVolume":    tx_volume,
    "HashRate":    hash_rate,
}, index=all_dates)
net_full.index.name = "Date"

# ── News ──────────────────────────────────────────────────────
news_templates = [
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

def make_news(dates, n_per_day=0.3):
    rows = []
    for d in dates:
        if np.random.rand() < n_per_day:
            idx_name = np.random.choice(["Index_A","Index_B","Index_C",
                                          "Index_D","Index_E","Index_F","markets"])
            tmpl = np.random.choice(news_templates)
            rows.append({"Date": d, "Headline": tmpl.format(idx=idx_name)})
    return pd.DataFrame(rows).set_index("Date")

train_news = make_news(train_dates, n_per_day=0.4)
test_news  = make_news(test_dates,  n_per_day=0.4)

# ── Guardar CSVs ─────────────────────────────────────────────
def save(df, name):
    path = os.path.join(OUT, name)
    df.to_csv(path)
    print(f"  {name:45s}  shape={str(df.shape):>15}  -> {path}")

print("Guardando datos ficticios en data/ ...")
save(train_idx,                          "train_indices.csv")
save(pd.DataFrame({"Date": test_dates}).set_index("Date"), "test_dates.csv")
save(macro_full.iloc[:N_TRAIN],          "train_macro_factors.csv")
save(macro_full.iloc[N_TRAIN:],          "test_macro_factors.csv")
save(net_full.iloc[:N_TRAIN],            "train_network_metrics.csv")
save(net_full.iloc[N_TRAIN:],            "test_network_metrics.csv")
save(train_news,                         "train_news.csv")
save(test_news,                          "test_news.csv")

print("\nVerificacion rapida:")
print(f"  Index_D vs Index_A lag-3 correlacion: "
      f"{train_idx['Index_D'].corr(train_idx['Index_A'].shift(3).dropna()):.4f}")
print("Listo.")
