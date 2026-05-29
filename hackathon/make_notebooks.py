"""Script that generates 03_LightGBM.ipynb, 04_LSTM.ipynb and 05_Ensemble.ipynb."""
import nbformat as nbf, os

BASE = r"c:\Users\1jose\Desktop\previa hackatlon\hackathon"

def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = cells
    return n

def md(src): return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)

def save(notebook, name):
    path = os.path.join(BASE, name)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(notebook, f)
    print(f"Saved {name}")


# ─────────────────────────────────────────────────────────
# 03_LightGBM.ipynb
# ─────────────────────────────────────────────────────────
lgbm_nb = nb([
md("""# 03 — LightGBM con Features de Lag
**Entrega 3.** Un modelo LightGBM por índice con lag features, predicción autorregresiva."""),

code("""\
import sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from utils import (load_data, compute_rmse, make_submission, train_val_split,
                   create_lag_features, add_calendar_features, add_log_returns,
                   find_ghost_source, INDEX_COLS)

data         = load_data()
train_full   = data["train_indices"][INDEX_COLS]
test_dates   = data["test_dates"].index
train, val   = train_val_split(train_full, val_size=252)
macro        = data.get("train_macro_factors")
network      = data.get("train_network_metrics")
macro_test   = data.get("test_macro_factors")
network_test = data.get("test_network_metrics")

LAGS    = (1, 2, 3, 5, 10, 20, 60)
WINDOWS = (5, 10, 20, 60)
print(f"Train: {train.shape}  |  Val: {val.shape}  |  Test: {len(test_dates)} días")
"""),

md("## Ghost detection — ¿qué índice imita Index_D?"),
code("find_ghost_source(train_full, target_col='Index_D', max_lag=30)"),

md("## Feature engineering"),
code("""\
def build_features(indices, macro=None, network=None):
    df = indices[INDEX_COLS].copy()
    df = add_log_returns(df)
    df = create_lag_features(df, lags=LAGS, windows=WINDOWS)
    df = add_calendar_features(df)
    if macro is not None:
        df = pd.concat([df, macro.reindex(df.index).ffill()], axis=1)
    if network is not None:
        df = pd.concat([df, network.reindex(df.index).ffill()], axis=1)
    return df

def prepare_xy(feats, targets):
    rename = {c: f"__t_{c}" for c in INDEX_COLS}
    combined = pd.concat([feats, targets.rename(columns=rename)], axis=1).dropna()
    tgt_cols = list(rename.values())
    return combined.drop(columns=tgt_cols).values, combined[tgt_cols].values, combined.index

macro_tr = macro.iloc[:-252] if macro is not None else None
net_tr   = network.iloc[:-252] if network is not None else None
feats_tr = build_features(train, macro_tr, net_tr)
X_tr, y_tr, _ = prepare_xy(feats_tr, train)
feature_names  = list(feats_tr.dropna().columns)
print(f"Feature matrix: {X_tr.shape}")
"""),

md("## Entrenamiento"),
code("""\
def train_lgbm(X, y, n_estimators=500):
    try:
        import lightgbm as lgb
        models = []
        for i, col in enumerate(INDEX_COLS):
            m = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=0.05,
                                  num_leaves=63, subsample=0.8,
                                  colsample_bytree=0.8, verbose=-1)
            m.fit(X, y[:, i])
            models.append(m)
        return models, "lgbm"
    except ImportError:
        pass
    try:
        from xgboost import XGBRegressor
        models = [XGBRegressor(n_estimators=n_estimators, learning_rate=0.05, verbosity=0
                               ).fit(X, y[:, i]) for i in range(y.shape[1])]
        return models, "xgb"
    except ImportError:
        pass
    from sklearn.ensemble import GradientBoostingRegressor
    models = [GradientBoostingRegressor(n_estimators=200).fit(X, y[:, i])
              for i in range(y.shape[1])]
    return models, "sklearn_gbr"

models, lib = train_lgbm(X_tr, y_tr)
print(f"Modelos entrenados: {lib}")
"""),

md("## Predicción autorregresiva"),
code("""\
def autoreg_predict(models, history, dates,
                    macro_all=None, net_all=None, feature_names=None):
    history = history[INDEX_COLS].copy()
    preds   = []
    for date in dates:
        feats = build_features(
            history,
            macro   = macro_all.loc[:date] if macro_all is not None else None,
            network = net_all.loc[:date]   if net_all   is not None else None,
        )
        row = feats.dropna().iloc[[-1]]
        if feature_names is not None:
            for c in feature_names:
                if c not in row.columns:
                    row[c] = 0.0
            row = row[feature_names]
        y_hat = np.array([m.predict(row.values)[0] for m in models])
        preds.append(y_hat)
        new_row = pd.DataFrame([y_hat], index=[date], columns=INDEX_COLS)
        history = pd.concat([history, new_row])
    return pd.DataFrame(preds, index=dates, columns=INDEX_COLS)
"""),

md("## Validación local"),
code("""\
pred_val = autoreg_predict(models, train, val.index,
                           macro_all=macro, net_all=network,
                           feature_names=feature_names)
rmse = compute_rmse(val, pred_val)
print(f"[LightGBM] RMSE local = {rmse:,.2f}")
per = np.sqrt(((val.values - pred_val.values)**2).mean(axis=0))
for col, r in zip(INDEX_COLS, per):
    print(f"  {col}: {r:,.2f}")
"""),

md("## Generar submission"),
code("""\
feats_full = build_features(train_full, macro, network)
X_full, y_full, _ = prepare_xy(feats_full, train_full)
fn_full = list(feats_full.dropna().columns)
models_full, _ = train_lgbm(X_full, y_full)

pred_test = autoreg_predict(models_full, train_full, test_dates,
                            macro_all=macro_test, net_all=network_test,
                            feature_names=fn_full)
make_submission(pred_test, "submission_03_lgbm.csv")
pred_test.head()
"""),
])
save(lgbm_nb, "03_LightGBM.ipynb")


# ─────────────────────────────────────────────────────────
# 04_LSTM.ipynb
# ─────────────────────────────────────────────────────────
lstm_nb = nb([
md("""# 04 — LSTM / Seq2Seq
**Entrega 4.** Red Seq2Seq: el encoder lee 60 días de historia, el decoder genera 252 pasos directamente.

Instalar PyTorch (CPU): `pip install torch --index-url https://download.pytorch.org/whl/cpu`"""),

code("""\
import sys, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from utils import load_data, compute_rmse, make_submission, train_val_split, INDEX_COLS

data       = load_data()
train_full = data["train_indices"][INDEX_COLS]
test_dates = data["test_dates"].index
train, val = train_val_split(train_full, val_size=252)
macro      = data.get("train_macro_factors")
network    = data.get("train_network_metrics")

WINDOW  = 60
HIDDEN  = 128
LAYERS  = 2
EPOCHS  = 50
BATCH   = 64
N_STEPS = 252
print("PyTorch version:", torch.__version__)
"""),

md("## Preparación de datos"),
code("""\
def prepare_data(indices, macro=None, network=None):
    df = indices[INDEX_COLS].copy()
    if macro   is not None: df = pd.concat([df, macro.reindex(df.index).ffill().fillna(0)],   axis=1)
    if network is not None: df = pd.concat([df, network.reindex(df.index).ffill().fillna(0)], axis=1)
    df = df.ffill().fillna(0)
    scaler_idx = StandardScaler()
    scaler_all = StandardScaler()
    scaler_idx.fit(df[INDEX_COLS].values)
    scaled     = scaler_all.fit_transform(df.values)
    return scaled, scaler_idx

def make_sequences(values, window, horizon):
    X, y = [], []
    for i in range(window, len(values) - horizon + 1):
        X.append(values[i - window:i])
        y.append(values[i:i + horizon, :6])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

macro_tr = macro.iloc[:-252] if macro is not None else None
net_tr   = network.iloc[:-252] if network is not None else None
scaled_tr, scaler_idx = prepare_data(train, macro_tr, net_tr)
X_tr, y_tr = make_sequences(scaled_tr, WINDOW, N_STEPS)
print(f"X: {X_tr.shape}  y: {y_tr.shape}")
"""),

md("## Arquitectura Seq2Seq"),
code("""\
class Encoder(nn.Module):
    def __init__(self, in_size, hidden, n_layers):
        super().__init__()
        self.lstm = nn.LSTM(in_size, hidden, n_layers,
                            batch_first=True, dropout=0.2)
    def forward(self, x):
        _, (h, c) = self.lstm(x)
        return h, c

class Decoder(nn.Module):
    def __init__(self, n_out, hidden, n_layers):
        super().__init__()
        self.lstm = nn.LSTM(n_out, hidden, n_layers,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden, n_out)
    def forward(self, h, c, steps):
        batch = h.shape[1]
        dec_in = torch.zeros(batch, 1, 6)
        outs = []
        for _ in range(steps):
            out, (h, c) = self.lstm(dec_in, (h, c))
            pred = self.fc(out)
            outs.append(pred)
            dec_in = pred
        return torch.cat(outs, dim=1)   # (batch, steps, 6)

n_features = X_tr.shape[2]
encoder = Encoder(n_features, HIDDEN, LAYERS)
decoder = Decoder(6, HIDDEN, LAYERS)
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
loss_fn = nn.MSELoss()
print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
print(f"Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")
"""),

md("## Entrenamiento"),
code("""\
Xt = torch.tensor(X_tr)
yt = torch.tensor(y_tr)

for ep in range(EPOCHS):
    idx = np.random.permutation(len(Xt))
    total = 0.0
    for s in range(0, len(Xt), BATCH):
        b = idx[s:s + BATCH]
        optimizer.zero_grad()
        h, c  = encoder(Xt[b])
        out   = decoder(h, c, N_STEPS)
        loss  = loss_fn(out, yt[b])
        loss.backward()
        optimizer.step()
        total += loss.item() * len(b)
    if (ep + 1) % 10 == 0:
        print(f"  Epoch {ep+1:>3}/{EPOCHS}  loss={total/len(Xt):.6f}")
"""),

md("## Validación local"),
code("""\
encoder.eval(); decoder.eval()
seed = torch.tensor(scaled_tr[-WINDOW:][None], dtype=torch.float32)
with torch.no_grad():
    h, c = encoder(seed)
    raw  = decoder(h, c, N_STEPS).numpy()[0]   # (252, 6)

preds_val = scaler_idx.inverse_transform(raw)
pred_df   = pd.DataFrame(preds_val, index=val.index, columns=INDEX_COLS)
rmse = compute_rmse(val, pred_df)
print(f"[Seq2Seq] RMSE local = {rmse:,.2f}")
per = np.sqrt(((val.values - pred_df.values)**2).mean(axis=0))
for col, r in zip(INDEX_COLS, per):
    print(f"  {col}: {r:,.2f}")
"""),

md("## Generar submission"),
code("""\
print("Entrenando en datos completos ...")
scaled_full, scaler_full_idx = prepare_data(train_full, macro, network)
X_full, y_full = make_sequences(scaled_full, WINDOW, N_STEPS)

enc_f = Encoder(X_full.shape[2], HIDDEN, LAYERS)
dec_f = Decoder(6, HIDDEN, LAYERS)
opt_f = torch.optim.Adam(
    list(enc_f.parameters()) + list(dec_f.parameters()), lr=1e-3)
Xf = torch.tensor(X_full); yf = torch.tensor(y_full)
for ep in range(EPOCHS):
    idx = np.random.permutation(len(Xf))
    for s in range(0, len(Xf), BATCH):
        b = idx[s:s + BATCH]
        opt_f.zero_grad()
        h, c = enc_f(Xf[b])
        loss_fn(dec_f(h, c, N_STEPS), yf[b]).backward()
        opt_f.step()
    if (ep + 1) % 10 == 0:
        print(f"  Epoch {ep+1}/{EPOCHS}")

enc_f.eval(); dec_f.eval()
seed_f = torch.tensor(scaled_full[-WINDOW:][None], dtype=torch.float32)
with torch.no_grad():
    h, c  = enc_f(seed_f)
    raw_f = dec_f(h, c, N_STEPS).numpy()[0]

preds_test = scaler_full_idx.inverse_transform(raw_f)
pred_test  = pd.DataFrame(preds_test, index=test_dates, columns=INDEX_COLS)
make_submission(pred_test, "submission_04_lstm_seq2seq.csv")
pred_test.head()
"""),
])
save(lstm_nb, "04_LSTM.ipynb")


# ─────────────────────────────────────────────────────────
# 05_Ensemble.ipynb
# ─────────────────────────────────────────────────────────
ens_nb = nb([
md("""# 05 — Ensemble + Corrección Index_D
**Entrega 5.** Combina baseline, ARIMA, LightGBM y LSTM con pesos optimizados.
Aplica corrección lineal a Index_D usando su índice fuente identificado en EDA."""),

code("""\
import sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from utils import load_data, compute_rmse, make_submission, train_val_split, find_ghost_source, INDEX_COLS

data       = load_data()
train_full = data["train_indices"][INDEX_COLS]
test_dates = data["test_dates"].index
train, val = train_val_split(train_full, val_size=252)
"""),

md("## Cargar predicciones individuales de validación"),
code("""\
SUBMISSIONS = "submissions"

def load_sub(fname, dates):
    df = pd.read_csv(f"{SUBMISSIONS}/{fname}", parse_dates=[0], index_col=0)
    return df.reindex(dates)[INDEX_COLS]

# Asegúrate de haber corrido los notebooks anteriores con datos de validación.
# Aquí cargamos los CSVs de test y re-evaluamos sobre val localmente.

# ----- Baseline -----
def rolling_mean(series_df, dates, window=20):
    m = series_df.tail(window).mean()
    return pd.DataFrame({c: [m[c]]*len(dates) for c in INDEX_COLS}, index=dates)

pred_baseline = rolling_mean(train, val.index, 20)
print(f"Baseline RMSE : {compute_rmse(val, pred_baseline):,.2f}")
"""),

md("## Re-generar predicciones val de LightGBM"),
code("""\
# Importamos las funciones directamente del notebook 03
from utils import create_lag_features, add_calendar_features, add_log_returns

macro   = data.get("train_macro_factors")
network = data.get("train_network_metrics")
macro_tr = macro.iloc[:-252] if macro is not None else None
net_tr   = network.iloc[:-252] if network is not None else None

LAGS = (1, 2, 3, 5, 10, 20, 60); WINDOWS = (5, 10, 20, 60)

def build_features(indices, macro=None, network=None):
    df = indices[INDEX_COLS].copy()
    df = add_log_returns(df)
    df = create_lag_features(df, lags=LAGS, windows=WINDOWS)
    df = add_calendar_features(df)
    if macro   is not None: df = pd.concat([df, macro.reindex(df.index).ffill()],   axis=1)
    if network is not None: df = pd.concat([df, network.reindex(df.index).ffill()], axis=1)
    return df

def prepare_xy(feats, targets):
    rename = {c: f"__t_{c}" for c in INDEX_COLS}
    combined = pd.concat([feats, targets.rename(columns=rename)], axis=1).dropna()
    tc = list(rename.values())
    return combined.drop(columns=tc).values, combined[tc].values, combined.index

def train_lgbm(X, y, n=500):
    try:
        import lightgbm as lgb
        ms = []
        for i in range(6):
            m = lgb.LGBMRegressor(n_estimators=n, learning_rate=0.05,
                                  num_leaves=63, subsample=0.8,
                                  colsample_bytree=0.8, verbose=-1)
            m.fit(X, y[:, i]); ms.append(m)
        return ms
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        return [GradientBoostingRegressor(n_estimators=200).fit(X, y[:, i]) for i in range(6)]

def autoreg_predict(models, history, dates, macro_all=None, net_all=None, fn=None):
    history = history[INDEX_COLS].copy(); preds = []
    for date in dates:
        feats = build_features(history,
                               macro   = macro_all.loc[:date] if macro_all is not None else None,
                               network = net_all.loc[:date]   if net_all   is not None else None)
        row = feats.dropna().iloc[[-1]]
        if fn is not None:
            for c in fn:
                if c not in row.columns: row[c] = 0.0
            row = row[fn]
        y_hat = np.array([m.predict(row.values)[0] for m in models])
        preds.append(y_hat)
        history = pd.concat([history, pd.DataFrame([y_hat], index=[date], columns=INDEX_COLS)])
    return pd.DataFrame(preds, index=dates, columns=INDEX_COLS)

feats_tr = build_features(train, macro_tr, net_tr)
X_tr, y_tr, _ = prepare_xy(feats_tr, train)
fn_tr = list(feats_tr.dropna().columns)
print("Entrenando LightGBM ...")
models_lgbm = train_lgbm(X_tr, y_tr)
pred_lgbm_val = autoreg_predict(models_lgbm, train, val.index,
                                macro_all=macro, net_all=network, fn=fn_tr)
print(f"LightGBM RMSE : {compute_rmse(val, pred_lgbm_val):,.2f}")
"""),

md("## Blend ponderado"),
code("""\
from itertools import product

all_preds_val = {"baseline": pred_baseline, "lgbm": pred_lgbm_val}

# Añade predicciones de ARIMA y LSTM si existen en submissions/
import os
for tag, fname in [("arima", "submission_02_arima.csv"),
                   ("lstm",  "submission_04_lstm_seq2seq.csv")]:
    p = os.path.join(SUBMISSIONS, fname)
    if os.path.exists(p):
        all_preds_val[tag] = pd.read_csv(p, parse_dates=[0], index_col=0).reindex(val.index)[INDEX_COLS]
        print(f"  Cargado {tag}: RMSE = {compute_rmse(val, all_preds_val[tag]):,.2f}")

names  = list(all_preds_val.keys())
stacked = np.stack([all_preds_val[n].values for n in names], axis=0)

best_w = np.ones(len(names)) / len(names)
best_r = np.inf

print("Buscando mejores pesos ...")
for combo in product(np.arange(0, 1.01, 0.1), repeat=len(names)):
    w = np.array(combo, dtype=float)
    if w.sum() < 1e-6: continue
    w /= w.sum()
    blended = np.einsum("i,ijk->jk", w, stacked)
    r = compute_rmse(val, pd.DataFrame(blended, index=val.index, columns=INDEX_COLS))
    if r < best_r:
        best_r, best_w = r, w.copy()

print(f"Mejor RMSE ensemble: {best_r:,.2f}")
print(f"Pesos: { {n: round(float(w),3) for n,w in zip(names, best_w)} }")
"""),

md("## Corrección de Index_D (The Ghost)"),
code("""\
from sklearn.linear_model import LinearRegression

source_col, lag, corr_val = find_ghost_source(train_full, target_col="Index_D", max_lag=30)

def ghost_correct(pred_df, train_full, ghost="Index_D"):
    sc, lag, cv = find_ghost_source(train_full, target_col=ghost, max_lag=30)
    if sc is None or abs(cv) < 0.8:
        print(f"  Correccion saltada (r={cv:.3f})")
        return pred_df
    src = train_full[sc].values
    tgt = train_full[ghost].values
    n   = len(src)
    if lag == 0:
        X_fit, y_fit = src.reshape(-1, 1), tgt
    else:
        X_fit, y_fit = src[:n-lag].reshape(-1, 1), tgt[lag:]
    lr = LinearRegression().fit(X_fit, y_fit)
    pred_c = pred_df.copy()
    if lag == 0:
        src_pred = pred_df[sc].values
    else:
        last_known = train_full[sc].values[-lag:]
        src_pred   = np.concatenate([last_known, pred_df[sc].values[:-lag]])
    pred_c[ghost] = lr.predict(src_pred.reshape(-1, 1))
    print(f"  Correccion aplicada: {ghost} ~ {sc}  r={cv:.3f}")
    return pred_c
"""),

md("## Generar submission final"),
code("""\
macro_test   = data.get("test_macro_factors")
network_test = data.get("test_network_metrics")

# Baseline test
pred_bl_test = rolling_mean(train_full, test_dates, 20)

# LightGBM test
feats_full  = build_features(train_full, macro, network)
X_f, y_f, _ = prepare_xy(feats_full, train_full)
fn_full     = list(feats_full.dropna().columns)
models_f    = train_lgbm(X_f, y_f)
pred_lgbm_test = autoreg_predict(models_f, train_full, test_dates,
                                 macro_all=macro_test, net_all=network_test, fn=fn_full)

test_preds_all = {"baseline": pred_bl_test, "lgbm": pred_lgbm_test}
for tag, fname in [("arima", "submission_02_arima.csv"),
                   ("lstm",  "submission_04_lstm_seq2seq.csv")]:
    p = os.path.join(SUBMISSIONS, fname)
    if os.path.exists(p):
        test_preds_all[tag] = pd.read_csv(p, parse_dates=[0], index_col=0).reindex(test_dates)[INDEX_COLS]

stacked_test = np.stack([test_preds_all[n].values for n in names if n in test_preds_all], axis=0)
w_use = np.array([best_w[i] for i, n in enumerate(names) if n in test_preds_all])
w_use /= w_use.sum()
blended_test = np.einsum("i,ijk->jk", w_use, stacked_test)
ensemble_test = pd.DataFrame(blended_test, index=test_dates, columns=INDEX_COLS)

ensemble_test = ghost_correct(ensemble_test, train_full)
make_submission(ensemble_test, "submission_05_ensemble.csv")
ensemble_test.head()
"""),
])
save(ens_nb, "05_Ensemble.ipynb")

print("\nTodos los notebooks generados correctamente.")
