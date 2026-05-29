"""Genera 07_HuggingFace.ipynb con modelos de HF:
   1. TimeSeriesTransformer  (transformers library)
   2. Chronos-T5-Small       (amazon/chronos-t5-small — zero-shot)
   3. Moirai                 (Salesforce/moirai — zero-shot)
"""
import nbformat as nbf, os

BASE = r"c:\Users\1jose\Desktop\previa hackatlon\hackathon"

def nb(cells):
    n = nbf.v4.new_notebook(); n.cells = cells; return n
def md(src):  return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)
def save(n, name):
    with open(os.path.join(BASE, name), "w", encoding="utf-8") as f:
        nbf.write(n, f)
    print(f"Saved {name}")


# ══════════════════════════════════════════════════════════════════
c0 = md("""\
# 07 — Modelos de Hugging Face para Forecasting

Tres enfoques con la librería `transformers` de HF:

| # | Modelo | Tipo | Ventaja |
|---|--------|------|---------|
| 1 | **TimeSeriesTransformer** | Fine-tunable | Arquitectura oficial HF para TS |
| 2 | **Chronos-T5-Small** (Amazon) | Zero-shot | No requiere entrenamiento |
| 3 | **Moirai** (Salesforce) | Zero-shot | Foundation model multi-variante |

### Instalación
```bash
pip install transformers accelerate
pip install git+https://github.com/amazon-science/chronos-forecasting.git
pip install uni2ts   # para Moirai
```
""")

# ══════════════════════════════════════════════════════════════════
c1 = code("""\
import sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from utils import load_data, compute_rmse, make_submission, train_val_split, INDEX_COLS

data       = load_data()
train_full = data["train_indices"][INDEX_COLS]
test_dates = data["test_dates"].index
train, val = train_val_split(train_full, val_size=252)

N_STEPS = 252
print(f"Train: {train.shape}  |  Val: {val.shape}  |  Test: {len(test_dates)}")
""")

# ──────────────────────────────────────────────────────────────────
# MODELO 1: TimeSeriesTransformer (HF)
# ──────────────────────────────────────────────────────────────────
c2 = md("""\
---
## Modelo 1 — `TimeSeriesTransformer` (Hugging Face `transformers`)

Modelo probabilístico que usa distribuciones de student-t como salida.
Entrenamos un modelo por índice (o uno multi-output con listas de series).

Docs: https://huggingface.co/docs/transformers/model_doc/time_series_transformer
""")

c3 = code("""\
from transformers import (
    TimeSeriesTransformerConfig,
    TimeSeriesTransformerForPrediction,
)
import torch
from torch.utils.data import DataLoader, TensorDataset

CONTEXT   = 60    # pasos de contexto
HORIZON   = 252
LAGS_SEQ  = [1, 2, 3, 5, 7, 10, 20, 30, 60]
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
""")

c4 = code("""\
def build_hf_dataset(series: np.ndarray, context: int, horizon: int):
    \"\"\"
    Construye tensores past_values, past_time_features, future_values, future_time_features.
    series: (T,) array de un solo índice (normalizado).
    \"\"\"
    past_vals, fut_vals = [], []
    for i in range(context, len(series) - horizon + 1):
        past_vals.append(series[i - context : i])
        fut_vals.append(series[i : i + horizon])
    past_vals = torch.tensor(np.array(past_vals), dtype=torch.float32)
    fut_vals  = torch.tensor(np.array(fut_vals),  dtype=torch.float32)
    # time features simples: posición normalizada
    T_past = torch.linspace(0, 1, context).unsqueeze(0).expand(len(past_vals), -1).unsqueeze(-1)
    T_fut  = torch.linspace(0, 1, horizon).unsqueeze(0).expand(len(fut_vals),  -1).unsqueeze(-1)
    return past_vals, T_past, fut_vals, T_fut
""")

c5 = code("""\
from sklearn.preprocessing import StandardScaler

def train_hf_tst(train_series: np.ndarray, col_name: str,
                 epochs=30, batch=16, lr=1e-4):
    sc = StandardScaler()
    scaled = sc.fit_transform(train_series.reshape(-1,1)).ravel()

    pv, tp, fv, tf = build_hf_dataset(scaled, CONTEXT, HORIZON)
    ds  = TensorDataset(pv, tp, fv, tf)
    dl  = DataLoader(ds, batch_size=batch, shuffle=True)

    cfg = TimeSeriesTransformerConfig(
        prediction_length      = HORIZON,
        context_length         = CONTEXT,
        lags_sequence          = LAGS_SEQ,
        num_time_features      = 1,
        num_dynamic_real_features = 0,
        num_static_categorical_features = 0,
        num_static_real_features = 0,
        d_model                = 64,
        encoder_layers         = 2,
        decoder_layers         = 2,
        encoder_attention_heads= 4,
        decoder_attention_heads= 4,
        encoder_ffn_dim        = 128,
        decoder_ffn_dim        = 128,
        dropout                = 0.1,
        distribution_output    = "student_t",
    )
    model = TimeSeriesTransformerForPrediction(cfg).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for ep in range(epochs):
        total = 0.0
        for pv_b, tp_b, fv_b, tf_b in dl:
            pv_b  = pv_b.to(device);  fv_b  = fv_b.to(device)
            tp_b  = tp_b.to(device);  tf_b  = tf_b.to(device)
            opt.zero_grad()
            out = model(
                past_values              = pv_b,
                past_time_features       = tp_b,
                future_values            = fv_b,
                future_time_features     = tf_b,
            )
            out.loss.backward()
            opt.step()
            total += out.loss.item() * len(pv_b)
        if (ep+1) % 10 == 0:
            print(f"  [{col_name}] Epoch {ep+1}/{epochs}  loss={total/len(ds):.5f}")

    # Inferencia: generar muestras y tomar la mediana
    model.eval()
    sc_last = torch.tensor(scaled[-CONTEXT:], dtype=torch.float32).unsqueeze(0).to(device)
    tp_inf  = torch.linspace(0, 1, CONTEXT).view(1, CONTEXT, 1).to(device)
    tf_inf  = torch.linspace(0, 1, HORIZON).view(1, HORIZON, 1).to(device)
    with torch.no_grad():
        out = model.generate(
            past_values         = sc_last,
            past_time_features  = tp_inf,
            future_time_features= tf_inf,
        )
    samples = out.sequences.cpu().numpy()      # (1, n_samples, horizon)
    median  = np.median(samples[0], axis=0)    # (horizon,)
    return sc.inverse_transform(median.reshape(-1,1)).ravel(), model, sc

print("Funciones HF TimeSeriesTransformer definidas.")
""")

c6 = code("""\
# Entrenar un modelo por cada índice
tst_val_preds = {}
tst_models    = {}

print("Entrenando TimeSeriesTransformer por índice (puede tardar ~3-5 min) ...")
for col in INDEX_COLS:
    print(f"\\n=== {col} ===")
    pred_arr, model, sc = train_hf_tst(train[col].values, col_name=col, epochs=20)
    tst_val_preds[col] = pred_arr
    tst_models[col]    = (model, sc)

pred_tst_val = pd.DataFrame(tst_val_preds, index=val.index)
rmse_tst = compute_rmse(val, pred_tst_val)
print(f"\\n[HF TimeSeriesTransformer] RMSE local = {rmse_tst:,.2f}")
per = np.sqrt(((val.values - pred_tst_val.values)**2).mean(axis=0))
for col, r in zip(INDEX_COLS, per): print(f"  {col}: {r:,.2f}")
""")

c7 = code("""\
# Submission con datos completos
tst_test_preds = {}
print("Reentrenando en datos completos ...")
for col in INDEX_COLS:
    print(f"  {col} ...", end=" ", flush=True)
    pred_arr, _, _ = train_hf_tst(train_full[col].values, col_name=col, epochs=20)
    tst_test_preds[col] = pred_arr
    print("ok")

pred_tst_test = pd.DataFrame(tst_test_preds, index=test_dates)
make_submission(pred_tst_test, "submission_07a_hf_tst.csv")
pred_tst_test.head()
""")

# ──────────────────────────────────────────────────────────────────
# MODELO 2: Chronos (Amazon) — zero-shot
# ──────────────────────────────────────────────────────────────────
c8 = md("""\
---
## Modelo 2 — Chronos-T5-Small (Amazon, zero-shot)

**Chronos** es un foundation model pre-entrenado sobre miles de datasets de series temporales.
No requiere fine-tuning: cargamos los pesos y predecimos directamente.

- Modelos disponibles: `amazon/chronos-t5-tiny`, `small`, `base`, `large`
- Paper: *Chronos: Learning the Language of Time Series* (Ansari et al. 2024)

```bash
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```
""")

c9 = code("""\
def forecast_chronos(series_dict: dict, test_dates, model_name="amazon/chronos-t5-small",
                     num_samples=20):
    \"\"\"
    series_dict: {col: pd.Series con datos históricos}
    Devuelve DataFrame (test_dates x INDEX_COLS) con la mediana de las muestras.
    \"\"\"
    try:
        from chronos import ChronosPipeline
    except ImportError:
        print("Chronos no instalado. Ejecuta:")
        print("  pip install git+https://github.com/amazon-science/chronos-forecasting.git")
        return None

    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    print(f"Modelo cargado: {model_name}")

    preds = {}
    for col, series in series_dict.items():
        context = torch.tensor(series.values, dtype=torch.float32)
        # generate devuelve (num_samples, prediction_length)
        forecast = pipeline.predict(
            context          = context,
            prediction_length= N_STEPS,
            num_samples      = num_samples,
        )
        median = np.median(forecast[0].numpy(), axis=0)  # (N_STEPS,)
        preds[col] = median
        print(f"  {col}: done")

    return pd.DataFrame(preds, index=test_dates)

# Validación (usa ultimos 252 días como pseudo-test)
print("Ejecutando Chronos en modo zero-shot (validación) ...")
train_series_val = {col: train[col] for col in INDEX_COLS}
pred_chronos_val = forecast_chronos(train_series_val, val.index)

if pred_chronos_val is not None:
    rmse_ch = compute_rmse(val, pred_chronos_val)
    print(f"\\n[Chronos zero-shot] RMSE local = {rmse_ch:,.2f}")
    per = np.sqrt(((val.values - pred_chronos_val.values)**2).mean(axis=0))
    for col, r in zip(INDEX_COLS, per): print(f"  {col}: {r:,.2f}")
""")

c10 = code("""\
# Submission Chronos (datos completos)
if pred_chronos_val is not None:
    train_series_full = {col: train_full[col] for col in INDEX_COLS}
    pred_chronos_test = forecast_chronos(train_series_full, test_dates)
    if pred_chronos_test is not None:
        make_submission(pred_chronos_test, "submission_07b_chronos.csv")
        pred_chronos_test.head()
""")

# ──────────────────────────────────────────────────────────────────
# MODELO 3: Moirai (Salesforce) — zero-shot
# ──────────────────────────────────────────────────────────────────
c11 = md("""\
---
## Modelo 3 — Moirai (Salesforce, zero-shot)

**MOIRAI** (Unified Training of Universal Time Series Forecasting Transformers)
es otro foundation model de Salesforce, diseñado para series multivariantes.

- Modelos: `Salesforce/moirai-1.0-R-small`, `base`, `large`
- Paper: *Unified Training of Universal TS Forecasting Transformers* (Woo et al. 2024)

```bash
pip install uni2ts
```
""")

c12 = code("""\
def forecast_moirai(series_df: pd.DataFrame, test_dates,
                    model_name="Salesforce/moirai-1.0-R-small",
                    num_samples=20):
    \"\"\"
    series_df: DataFrame (T x 6) con los datos de entrenamiento.
    \"\"\"
    try:
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        import gluonts
        from gluonts.dataset.pandas import PandasDataset
    except ImportError:
        print("Moirai/uni2ts no instalado. Ejecuta: pip install uni2ts")
        return None

    # Convertir a formato GluonTS
    dataset = PandasDataset(dict(series_df), freq="B")  # B = business days

    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(model_name),
        prediction_length = N_STEPS,
        context_length    = 200,
        patch_size        = "auto",
        num_samples       = num_samples,
        target_dim        = 1,
        feat_dynamic_real_dim = 0,
        past_feat_dynamic_real_dim = 0,
    )

    predictor = model.create_predictor(batch_size=8)
    forecasts  = list(predictor.predict(dataset))

    preds = {}
    for fc, col in zip(forecasts, INDEX_COLS):
        preds[col] = np.median(fc.samples, axis=0)  # (N_STEPS,)
        print(f"  {col}: done")

    return pd.DataFrame(preds, index=test_dates)


print("Ejecutando Moirai en modo zero-shot (validación) ...")
pred_moirai_val = forecast_moirai(train, val.index)

if pred_moirai_val is not None:
    rmse_m = compute_rmse(val, pred_moirai_val)
    print(f"\\n[Moirai zero-shot] RMSE local = {rmse_m:,.2f}")
    per = np.sqrt(((val.values - pred_moirai_val.values)**2).mean(axis=0))
    for col, r in zip(INDEX_COLS, per): print(f"  {col}: {r:,.2f}")
""")

c13 = code("""\
if pred_moirai_val is not None:
    pred_moirai_test = forecast_moirai(train_full, test_dates)
    if pred_moirai_test is not None:
        make_submission(pred_moirai_test, "submission_07c_moirai.csv")
        pred_moirai_test.head()
""")

# ──────────────────────────────────────────────────────────────────
# Comparativa final + Mega-ensemble
# ──────────────────────────────────────────────────────────────────
c14 = md("---\n## Comparativa global y Mega-Ensemble")

c15 = code("""\
import os

all_results = {}

# Recoger RMSE de modelos disponibles
for tag, pred in [
    ("HF_TST",   pred_tst_val   if 'pred_tst_val'   in dir() else None),
    ("Chronos",  pred_chronos_val if 'pred_chronos_val' in dir() and pred_chronos_val is not None else None),
    ("Moirai",   pred_moirai_val  if 'pred_moirai_val'  in dir() and pred_moirai_val  is not None else None),
]:
    if pred is not None:
        all_results[tag] = compute_rmse(val, pred)

df_cmp = pd.DataFrame.from_dict(all_results, orient="index", columns=["RMSE_local"]).sort_values("RMSE_local")
print("\\n=== Comparativa modelos HF ===")
print(df_cmp.round(2))
""")

c16 = code("""\
# Mega-ensemble: mezcla con todos los submissions disponibles
candidates = {}
for fname in os.listdir("submissions"):
    if fname.endswith(".csv"):
        df = pd.read_csv(f"submissions/{fname}", parse_dates=[0], index_col=0)
        candidates[fname] = df.reindex(val.index)[INDEX_COLS]

if candidates:
    stack  = np.stack([v.values for v in candidates.values()], axis=0)
    mega   = stack.mean(axis=0)
    pred_mega = pd.DataFrame(mega, index=val.index, columns=INDEX_COLS)
    rmse_mega = compute_rmse(val, pred_mega)
    print(f"\\n[Mega-Ensemble ({len(candidates)} modelos)] RMSE local = {rmse_mega:,.2f}")
    print("Modelos incluidos:", list(candidates.keys()))
""")

c17 = code("""\
# Submission mega-ensemble con datos de test
test_candidates = {}
for fname in os.listdir("submissions"):
    if fname.endswith(".csv"):
        df = pd.read_csv(f"submissions/{fname}", parse_dates=[0], index_col=0)
        df_reindexed = df.reindex(test_dates)[INDEX_COLS]
        if not df_reindexed.isnull().any().any():
            test_candidates[fname] = df_reindexed

if test_candidates:
    stack_test = np.stack([v.values for v in test_candidates.values()], axis=0)
    mega_test  = stack_test.mean(axis=0)
    pred_mega_test = pd.DataFrame(mega_test, index=test_dates, columns=INDEX_COLS)
    make_submission(pred_mega_test, "submission_08_mega_ensemble.csv")
    print(f"\\nMega-ensemble guardado con {len(test_candidates)} modelos.")
    pred_mega_test.head()
""")

# ══════════════════════════════════════════════════════════════════
hf_nb = nb([c0, c1, c2, c3, c4, c5, c6, c7,
            c8, c9, c10, c11, c12, c13,
            c14, c15, c16, c17])
save(hf_nb, "07_HuggingFace.ipynb")
print("Listo.")
