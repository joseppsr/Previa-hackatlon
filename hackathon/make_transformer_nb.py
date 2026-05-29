"""Genera 06_Transformers.ipynb con tres arquitecturas:
   1. Vanilla Transformer (encoder-only, autorregresivo)
   2. Temporal Fusion Transformer (TFT) simplificado
   3. PatchTST (parches de series temporales)
"""
import nbformat as nbf, os

BASE = r"c:\Users\1jose\Desktop\previa hackatlon\hackathon"

def nb(cells):
    n = nbf.v4.new_notebook(); n.cells = cells; return n
def md(src): return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)
def save(n, name):
    with open(os.path.join(BASE, name), "w", encoding="utf-8") as f:
        nbf.write(n, f)
    print(f"Saved {name}")


# ══════════════════════════════════════════════════════════════════
# CELDA 0 – cabecera
# ══════════════════════════════════════════════════════════════════
c0 = md("""# 06 — Modelos Transformer para Forecasting
**Entrega 6 (o mejora de la 5).** Tres arquitecturas:

| Modelo | Idea clave |
|--------|-----------|
| **Vanilla Transformer** | Self-attention sobre ventana temporal + decoder autorregresivo |
| **TFT** (Temporal Fusion Transformer) | Gating + Variable Selection + Multi-head attention |
| **PatchTST** | Divide la serie en *patches* → cada patch = token → Transformer encoder |

Instalar (si no está):
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
""")

# ══════════════════════════════════════════════════════════════════
# CELDA 1 – imports y carga de datos
# ══════════════════════════════════════════════════════════════════
c1 = code("""\
import sys, warnings, math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from utils import load_data, compute_rmse, make_submission, train_val_split, INDEX_COLS

data       = load_data()
train_full = data["train_indices"][INDEX_COLS]
test_dates = data["test_dates"].index
train, val = train_val_split(train_full, val_size=252)
macro      = data.get("train_macro_factors")
network    = data.get("train_network_metrics")

WINDOW  = 60     # longitud de la ventana de entrada
N_STEPS = 252    # días a predecir
N_IDX   = 6      # número de índices
EPOCHS  = 60
BATCH   = 32
LR      = 5e-4
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  |  Train: {train.shape}  |  Val: {val.shape}")
""")

# ══════════════════════════════════════════════════════════════════
# CELDA 2 – preparación de datos compartida
# ══════════════════════════════════════════════════════════════════
c2 = code("""\
def prepare_data(indices, macro=None, network=None):
    \"\"\"Escala y concatena todas las features. Devuelve (array_scaled, scaler_indices).\"\"\"
    df = indices[INDEX_COLS].copy()
    if macro   is not None: df = pd.concat([df, macro.reindex(df.index).ffill().fillna(0)],   axis=1)
    if network is not None: df = pd.concat([df, network.reindex(df.index).ffill().fillna(0)], axis=1)
    df = df.ffill().fillna(0)
    sc_idx = StandardScaler().fit(df[INDEX_COLS].values)
    sc_all = StandardScaler()
    scaled = sc_all.fit_transform(df.values).astype(np.float32)
    return scaled, sc_idx

def make_sequences(values, window, horizon):
    \"\"\"Devuelve X (N, window, F) e y (N, horizon, 6).\"\"\"
    X, y = [], []
    for i in range(window, len(values) - horizon + 1):
        X.append(values[i - window : i])
        y.append(values[i : i + horizon, :N_IDX])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train_loop(model, optimizer, Xt, yt, epochs=EPOCHS, batch=BATCH):
    loss_fn = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        idx = np.random.permutation(len(Xt))
        total = 0.0
        for s in range(0, len(Xt), batch):
            b  = idx[s:s+batch]
            xb = Xt[b].to(device)
            yb = yt[b].to(device)
            optimizer.zero_grad()
            out  = model(xb)               # (B, horizon, 6)  o  (B, 6)
            loss = loss_fn(out, yb)
            loss.backward(); optimizer.step()
            total += loss.item() * len(b)
        if (ep+1) % 10 == 0:
            print(f"  Epoch {ep+1:>3}/{epochs}  loss={total/len(Xt):.6f}")

macro_tr = macro.iloc[:-252] if macro is not None else None
net_tr   = network.iloc[:-252] if network is not None else None
scaled_tr, scaler_idx_tr = prepare_data(train, macro_tr, net_tr)
X_tr, y_tr = make_sequences(scaled_tr, WINDOW, N_STEPS)
print(f"Sequences  X: {X_tr.shape}   y: {y_tr.shape}")
""")

# ══════════════════════════════════════════════════════════════════
# CELDA 3 – MODELO 1: Vanilla Transformer (encoder-only Seq2Seq)
# ══════════════════════════════════════════════════════════════════
c3 = md("---\n## Modelo 1 — Vanilla Transformer\nUsa un encoder Transformer sobre la ventana y un decoder lineal que genera los 252 pasos de golpe.")

c4 = code("""\
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class VanillaTransformer(nn.Module):
    \"\"\"
    Encoder-only Transformer:
      (B, window, n_feat) -> proyecto a d_model -> N capas self-attention
      -> tomo el token [CLS] (primer paso) -> lineal a (horizon * 6)
    \"\"\"
    def __init__(self, n_feat, d_model=128, nhead=4, num_layers=3,
                 dim_ff=256, dropout=0.1, horizon=N_STEPS, n_out=N_IDX):
        super().__init__()
        self.input_proj = nn.Linear(n_feat, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff,
                                           dropout, batch_first=True)
        self.encoder    = nn.TransformerEncoder(layer, num_layers)
        self.head       = nn.Linear(d_model, horizon * n_out)
        self.horizon    = horizon
        self.n_out      = n_out

    def forward(self, x):
        # x: (B, window, n_feat)
        x = self.pos_enc(self.input_proj(x))    # (B, window, d_model)
        x = self.encoder(x)                     # (B, window, d_model)
        cls = x[:, -1, :]                       # último token como resumen
        out = self.head(cls)                    # (B, horizon*n_out)
        return out.view(-1, self.horizon, self.n_out)
""")

c5 = code("""\
n_feat = X_tr.shape[2]
vt_model = VanillaTransformer(n_feat).to(device)
vt_opt   = torch.optim.AdamW(vt_model.parameters(), lr=LR, weight_decay=1e-4)

Xt = torch.tensor(X_tr); yt = torch.tensor(y_tr)
print("Entrenando Vanilla Transformer ...")
train_loop(vt_model, vt_opt, Xt, yt)
""")

c6 = code("""\
# Validación
vt_model.eval()
with torch.no_grad():
    seed = torch.tensor(scaled_tr[-WINDOW:][None]).to(device)
    raw  = vt_model(seed).cpu().numpy()[0]   # (252, 6)

preds_vt = scaler_idx_tr.inverse_transform(raw)
pred_vt  = pd.DataFrame(preds_vt, index=val.index, columns=INDEX_COLS)
rmse_vt  = compute_rmse(val, pred_vt)
print(f"[Vanilla Transformer] RMSE local = {rmse_vt:,.2f}")
per = np.sqrt(((val.values - pred_vt.values)**2).mean(axis=0))
for col, r in zip(INDEX_COLS, per): print(f"  {col}: {r:,.2f}")
""")

# ══════════════════════════════════════════════════════════════════
# CELDA – MODELO 2: TFT simplificado
# ══════════════════════════════════════════════════════════════════
c7 = md("""---
## Modelo 2 — Temporal Fusion Transformer (TFT simplificado)
Bloques clave del TFT original (Lim et al. 2020):
- **Variable Selection Network**: pondera la importancia de cada feature con softmax gating
- **Gated Residual Network (GRN)**: proyección no lineal con gate ELU
- **Multi-Head Attention** sobre la ventana temporal
- Salida directa de los 252 pasos

Simplificamos la separación static/dynamic para que funcione sin metadata adicional.""")

c8 = code("""\
class GRN(nn.Module):
    \"\"\"Gated Residual Network.\"\"\"
    def __init__(self, d):
        super().__init__()
        self.fc1  = nn.Linear(d, d)
        self.fc2  = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.ln   = nn.LayerNorm(d)

    def forward(self, x):
        h     = F.elu(self.fc1(x))
        h     = self.fc2(h)
        g     = torch.sigmoid(self.gate(x))
        return self.ln(x + g * h)


class VariableSelectionNetwork(nn.Module):
    \"\"\"Selecciona y pondera cada feature con un gate suave.\"\"\"
    def __init__(self, n_feat, d_model):
        super().__init__()
        self.proj  = nn.Linear(n_feat, d_model)
        self.grn   = GRN(d_model)
        self.softmax_gate = nn.Sequential(
            nn.Linear(n_feat, n_feat), nn.Softmax(dim=-1))

    def forward(self, x):
        # x: (B, T, n_feat)
        weights = self.softmax_gate(x)          # (B, T, n_feat)
        x_w     = x * weights                   # feature weighting
        return self.grn(self.proj(x_w))         # (B, T, d_model)


class TFT(nn.Module):
    def __init__(self, n_feat, d_model=128, nhead=4, num_layers=2,
                 horizon=N_STEPS, n_out=N_IDX, dropout=0.1):
        super().__init__()
        self.vsn     = VariableSelectionNetwork(n_feat, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2,
                                           dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.grn_out = GRN(d_model)
        self.head    = nn.Linear(d_model, horizon * n_out)
        self.horizon = horizon; self.n_out = n_out

    def forward(self, x):
        x = self.pos_enc(self.vsn(x))       # (B, T, d_model)
        x = self.encoder(x)                 # (B, T, d_model)
        x = self.grn_out(x[:, -1, :])      # (B, d_model)
        return self.head(x).view(-1, self.horizon, self.n_out)
""")

c9 = code("""\
tft_model = TFT(n_feat).to(device)
tft_opt   = torch.optim.AdamW(tft_model.parameters(), lr=LR, weight_decay=1e-4)
print("Entrenando TFT ...")
train_loop(tft_model, tft_opt, Xt, yt)
""")

c10 = code("""\
tft_model.eval()
with torch.no_grad():
    seed = torch.tensor(scaled_tr[-WINDOW:][None]).to(device)
    raw  = tft_model(seed).cpu().numpy()[0]

preds_tft = scaler_idx_tr.inverse_transform(raw)
pred_tft  = pd.DataFrame(preds_tft, index=val.index, columns=INDEX_COLS)
rmse_tft  = compute_rmse(val, pred_tft)
print(f"[TFT] RMSE local = {rmse_tft:,.2f}")
per = np.sqrt(((val.values - pred_tft.values)**2).mean(axis=0))
for col, r in zip(INDEX_COLS, per): print(f"  {col}: {r:,.2f}")
""")

# ══════════════════════════════════════════════════════════════════
# CELDA – MODELO 3: PatchTST
# ══════════════════════════════════════════════════════════════════
c11 = md("""---
## Modelo 3 — PatchTST
*(Nie et al. 2023 — "A Time Series is Worth 64 Words")*

**Idea:** divide la ventana temporal en *patches* solapados → cada patch = un token → Transformer encoder.
- Reduce la longitud de la secuencia (menos atención cuadrática)
- Captura patrones locales en cada patch

Aquí hacemos la versión **Channel-Independent**: un Transformer por índice, compartiendo pesos.""")

c12 = code("""\
class PatchTST(nn.Module):
    \"\"\"
    Channel-Independent PatchTST.
    Cada uno de los N_IDX canales pasa por el mismo Transformer de forma independiente.
    \"\"\"
    def __init__(self, window=WINDOW, patch_len=12, stride=6,
                 d_model=128, nhead=4, num_layers=3, dropout=0.1,
                 horizon=N_STEPS, n_channels=N_IDX):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        n_patches      = (window - patch_len) // stride + 1

        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_len=n_patches + 1, dropout=dropout)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2,
                                           dropout, batch_first=True)
        self.encoder    = nn.TransformerEncoder(layer, num_layers)
        self.head       = nn.Linear(d_model * n_patches, horizon)
        self.n_patches  = n_patches
        self.horizon    = horizon
        self.n_channels = n_channels

    def _patchify(self, x):
        \"\"\"x: (B, T)  ->  (B, n_patches, patch_len)\"\"\"
        patches = []
        for i in range(0, len(range(0, x.size(1) - self.patch_len + 1, self.stride))):
            start = i * self.stride
            patches.append(x[:, start : start + self.patch_len])
        return torch.stack(patches, dim=1)   # (B, n_patches, patch_len)

    def forward(self, x):
        # x: (B, window, n_feat)  — usamos sólo los primeros n_channels
        B = x.size(0)
        outs = []
        for ch in range(self.n_channels):
            xc = x[:, :, ch]                        # (B, window)
            p  = self._patchify(xc)                 # (B, n_patches, patch_len)
            p  = self.pos_enc(self.patch_proj(p))   # (B, n_patches, d_model)
            p  = self.encoder(p)                    # (B, n_patches, d_model)
            p  = p.reshape(B, -1)                   # (B, n_patches * d_model)
            outs.append(self.head(p))               # (B, horizon)
        return torch.stack(outs, dim=2)             # (B, horizon, n_channels)
""")

c13 = code("""\
patch_model = PatchTST().to(device)
patch_opt   = torch.optim.AdamW(patch_model.parameters(), lr=LR, weight_decay=1e-4)
print("Entrenando PatchTST ...")
train_loop(patch_model, patch_opt, Xt, yt)
""")

c14 = code("""\
patch_model.eval()
with torch.no_grad():
    seed = torch.tensor(scaled_tr[-WINDOW:][None]).to(device)
    raw  = patch_model(seed).cpu().numpy()[0]   # (252, 6)

preds_patch = scaler_idx_tr.inverse_transform(raw)
pred_patch  = pd.DataFrame(preds_patch, index=val.index, columns=INDEX_COLS)
rmse_patch  = compute_rmse(val, pred_patch)
print(f"[PatchTST] RMSE local = {rmse_patch:,.2f}")
per = np.sqrt(((val.values - pred_patch.values)**2).mean(axis=0))
for col, r in zip(INDEX_COLS, per): print(f"  {col}: {r:,.2f}")
""")

# ══════════════════════════════════════════════════════════════════
# CELDA – Comparativa y mejor modelo
# ══════════════════════════════════════════════════════════════════
c15 = md("---\n## Comparativa y selección del mejor modelo")

c16 = code("""\
results = {
    "VanillaTransformer": rmse_vt,
    "TFT":                rmse_tft,
    "PatchTST":           rmse_patch,
}
df_res = pd.DataFrame.from_dict(results, orient="index", columns=["RMSE_local"])
df_res = df_res.sort_values("RMSE_local")
print(df_res.round(2))
best_name = df_res.index[0]
print(f"\\nMejor modelo: {best_name}  (RMSE={df_res.iloc[0,0]:,.2f})")
""")

# ══════════════════════════════════════════════════════════════════
# CELDA – Ensemble Transformers
# ══════════════════════════════════════════════════════════════════
c17 = md("## Ensemble de los tres Transformers")

c18 = code("""\
# Promedio simple de los tres modelos Transformer
blended = (pred_vt.values + pred_tft.values + pred_patch.values) / 3
pred_blend = pd.DataFrame(blended, index=val.index, columns=INDEX_COLS)
rmse_blend = compute_rmse(val, pred_blend)
print(f"[Ensemble Transformers] RMSE local = {rmse_blend:,.2f}")
""")

# ══════════════════════════════════════════════════════════════════
# CELDA – Generar submission con el mejor modelo
# ══════════════════════════════════════════════════════════════════
c19 = md("## Generar submission — entrenar en datos completos")

c20 = code("""\
scaled_full, scaler_full_idx = prepare_data(train_full, macro, network)
X_full, y_full = make_sequences(scaled_full, WINDOW, N_STEPS)
Xf = torch.tensor(X_full); yf = torch.tensor(y_full)

def retrain_and_predict(ModelClass, kwargs, Xf, yf, scaled_full):
    m   = ModelClass(**kwargs).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    train_loop(m, opt, Xf, yf, epochs=EPOCHS)
    m.eval()
    with torch.no_grad():
        seed = torch.tensor(scaled_full[-WINDOW:][None]).to(device)
        raw  = m(seed).cpu().numpy()[0]
    return scaler_full_idx.inverse_transform(raw)

n_feat_full = X_full.shape[2]
print(f"n_features (full): {n_feat_full}")

print("\\n--- Reentrenando Vanilla Transformer ---")
arr_vt   = retrain_and_predict(VanillaTransformer, {"n_feat": n_feat_full}, Xf, yf, scaled_full)

print("\\n--- Reentrenando TFT ---")
arr_tft  = retrain_and_predict(TFT, {"n_feat": n_feat_full}, Xf, yf, scaled_full)

print("\\n--- Reentrenando PatchTST ---")
arr_patch = retrain_and_predict(PatchTST, {}, Xf, yf, scaled_full)

# Ensemble final
arr_ens = (arr_vt + arr_tft + arr_patch) / 3
pred_test = pd.DataFrame(arr_ens, index=test_dates, columns=INDEX_COLS)
make_submission(pred_test, "submission_06_transformers.csv")
print("\\nSubmission generada: submission_06_transformers.csv")
pred_test.head()
""")

# ══════════════════════════════════════════════════════════════════
# CELDA – Guardar pesos
# ══════════════════════════════════════════════════════════════════
c21 = md("## (Opcional) Guardar y cargar pesos")

c22 = code("""\
import os
os.makedirs("checkpoints", exist_ok=True)

# Guardar  (ejecutar tras entrenar en datos completos)
# torch.save(m.state_dict(), "checkpoints/patchtst_full.pt")

# Cargar
# m = PatchTST().to(device)
# m.load_state_dict(torch.load("checkpoints/patchtst_full.pt", map_location=device))
print("Bloque de guardado listo (comentado por defecto).")
""")

# ══════════════════════════════════════════════════════════════════
# Ensamblar notebook y guardar
# ══════════════════════════════════════════════════════════════════
transformer_nb = nb([
    c0, c1, c2,
    c3, c4, c5, c6,
    c7, c8, c9, c10,
    c11, c12, c13, c14,
    c15, c16,
    c17, c18,
    c19, c20,
    c21, c22,
])
save(transformer_nb, "06_Transformers.ipynb")
print("Listo.")
