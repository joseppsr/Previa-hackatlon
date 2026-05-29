"""Regenera 06_Transformers.ipynb con todas las constantes definidas ANTES de las clases."""
import nbformat as nbf, os

BASE = r"c:\Users\1jose\Desktop\previa hackatlon\hackathon"

def nb(cells): n = nbf.v4.new_notebook(); n.cells = cells; return n
def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)
def save(notebook, name):
    with open(os.path.join(BASE, name), "w", encoding="utf-8") as f:
        nbf.write(notebook, f)
    print(f"Saved {name}")

# ── celda 1: setup + constantes + carga de datos ──────────────────
C_SETUP = """\
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

# Hiperparametros  (CPU: valores actuales;  GPU: EPOCHS=60, D_MODEL=128, NHEAD=4, NLAYERS=3)
WINDOW  = 60
HORIZON = 252
N_IDX   = 6
EPOCHS  = 10
BATCH   = 32
LR      = 5e-4
D_MODEL = 64
NHEAD   = 2
NLAYERS = 1
DIM_FF  = 128
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, " | Train:", train.shape, " | Val:", val.shape)
"""

# ── celda 2: preparacion de datos ─────────────────────────────────
C_DATA = """\
def prepare_data(indices, macro=None, network=None):
    df = indices[INDEX_COLS].copy()
    if macro   is not None: df = pd.concat([df, macro.reindex(df.index).ffill().fillna(0)],   axis=1)
    if network is not None: df = pd.concat([df, network.reindex(df.index).ffill().fillna(0)], axis=1)
    df = df.ffill().fillna(0)
    sc_idx = StandardScaler().fit(df[INDEX_COLS].values)
    sc_all = StandardScaler()
    scaled = sc_all.fit_transform(df.values).astype("float32")
    return scaled, sc_idx

def make_sequences(values, window, horizon):
    X, y = [], []
    for i in range(window, len(values) - horizon + 1):
        X.append(values[i - window : i])
        y.append(values[i : i + horizon, :N_IDX])
    return np.array(X, "float32"), np.array(y, "float32")

def train_loop(model, opt, Xt, yt, epochs=EPOCHS, batch=BATCH):
    loss_fn = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        idx = np.random.permutation(len(Xt))
        total = 0.0
        for s in range(0, len(Xt), batch):
            b = idx[s:s+batch]
            xb, yb = Xt[b].to(device), yt[b].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
            total += loss.item() * len(b)
        if (ep+1) % max(1, epochs//3) == 0:
            print(f"  Epoch {ep+1}/{epochs}  loss={total/len(Xt):.6f}")

macro_tr = macro.iloc[:-252] if macro is not None else None
net_tr   = network.iloc[:-252] if network is not None else None
scaled_tr, scaler_idx_tr = prepare_data(train, macro_tr, net_tr)
X_tr, y_tr = make_sequences(scaled_tr, WINDOW, HORIZON)
Xt = torch.tensor(X_tr); yt = torch.tensor(y_tr)
n_feat = X_tr.shape[2]
print(f"Sequences X:{X_tr.shape}  y:{y_tr.shape}  n_feat={n_feat}")
"""

# ── celda 3: PositionalEncoding ────────────────────────────────────
C_PE = """\
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])
"""

# ── Vanilla Transformer ────────────────────────────────────────────
C_VT = """\
class VanillaTransformer(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.input_proj = nn.Linear(n_feat, D_MODEL)
        self.pos_enc    = PositionalEncoding(D_MODEL)
        layer = nn.TransformerEncoderLayer(D_MODEL, NHEAD, DIM_FF, batch_first=True)
        self.encoder    = nn.TransformerEncoder(layer, NLAYERS)
        self.head       = nn.Linear(D_MODEL, HORIZON * N_IDX)
    def forward(self, x):
        x = self.pos_enc(self.input_proj(x))
        x = self.encoder(x)
        return self.head(x[:, -1, :]).view(-1, HORIZON, N_IDX)

vt = VanillaTransformer(n_feat).to(device)
print("VanillaTransformer params:", sum(p.numel() for p in vt.parameters()))
train_loop(vt, torch.optim.AdamW(vt.parameters(), lr=LR), Xt, yt)
"""

C_VT_VAL = """\
vt.eval()
with torch.no_grad():
    raw = vt(torch.tensor(scaled_tr[-WINDOW:][None]).to(device)).cpu().numpy()[0]
pred_vt = pd.DataFrame(scaler_idx_tr.inverse_transform(raw), index=val.index, columns=INDEX_COLS)
rmse_vt = compute_rmse(val, pred_vt)
print(f"[VanillaTransformer] RMSE = {rmse_vt:,.2f}")
per = np.sqrt(((val.values - pred_vt.values)**2).mean(axis=0))
for col, r in zip(INDEX_COLS, per): print(f"  {col}: {r:,.2f}")
"""

# ── TFT ───────────────────────────────────────────────────────────
C_TFT = """\
class GRN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1  = nn.Linear(d, d)
        self.fc2  = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.ln   = nn.LayerNorm(d)
    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = self.fc2(h)
        g = torch.sigmoid(self.gate(x))
        return self.ln(x + g * h)

class VSN(nn.Module):
    def __init__(self, n_feat, d):
        super().__init__()
        self.proj   = nn.Linear(n_feat, d)
        self.grn    = GRN(d)
        self.gating = nn.Sequential(nn.Linear(n_feat, n_feat), nn.Softmax(dim=-1))
    def forward(self, x):
        return self.grn(self.proj(x * self.gating(x)))

class TFT(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.vsn     = VSN(n_feat, D_MODEL)
        self.pos_enc = PositionalEncoding(D_MODEL)
        layer = nn.TransformerEncoderLayer(D_MODEL, NHEAD, D_MODEL*2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, NLAYERS)
        self.grn_out = GRN(D_MODEL)
        self.head    = nn.Linear(D_MODEL, HORIZON * N_IDX)
    def forward(self, x):
        x = self.pos_enc(self.vsn(x))
        x = self.encoder(x)
        return self.head(self.grn_out(x[:, -1, :])).view(-1, HORIZON, N_IDX)

tft = TFT(n_feat).to(device)
print("TFT params:", sum(p.numel() for p in tft.parameters()))
train_loop(tft, torch.optim.AdamW(tft.parameters(), lr=LR), Xt, yt)
"""

C_TFT_VAL = """\
tft.eval()
with torch.no_grad():
    raw = tft(torch.tensor(scaled_tr[-WINDOW:][None]).to(device)).cpu().numpy()[0]
pred_tft = pd.DataFrame(scaler_idx_tr.inverse_transform(raw), index=val.index, columns=INDEX_COLS)
rmse_tft = compute_rmse(val, pred_tft)
print(f"[TFT] RMSE = {rmse_tft:,.2f}")
per = np.sqrt(((val.values - pred_tft.values)**2).mean(axis=0))
for col, r in zip(INDEX_COLS, per): print(f"  {col}: {r:,.2f}")
"""

# ── PatchTST ──────────────────────────────────────────────────────
C_PATCH = """\
PATCH_LEN = 12
STRIDE    = 6
N_PATCHES = (WINDOW - PATCH_LEN) // STRIDE + 1

class PatchTST(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_proj = nn.Linear(PATCH_LEN, D_MODEL)
        self.pos_enc    = PositionalEncoding(D_MODEL, max_len=N_PATCHES + 1)
        layer = nn.TransformerEncoderLayer(D_MODEL, NHEAD, D_MODEL*2, batch_first=True)
        self.encoder    = nn.TransformerEncoder(layer, NLAYERS)
        self.head       = nn.Linear(D_MODEL * N_PATCHES, HORIZON)

    def _patch(self, x):
        return torch.stack([x[:, i*STRIDE : i*STRIDE + PATCH_LEN] for i in range(N_PATCHES)], dim=1)

    def forward(self, x):
        B = x.size(0)
        outs = []
        for ch in range(N_IDX):
            p = self.pos_enc(self.patch_proj(self._patch(x[:, :, ch])))
            p = self.encoder(p).reshape(B, -1)
            outs.append(self.head(p))
        return torch.stack(outs, dim=2)   # (B, HORIZON, N_IDX)

patch = PatchTST().to(device)
print("PatchTST params:", sum(p.numel() for p in patch.parameters()))
train_loop(patch, torch.optim.AdamW(patch.parameters(), lr=LR), Xt, yt)
"""

C_PATCH_VAL = """\
patch.eval()
with torch.no_grad():
    raw = patch(torch.tensor(scaled_tr[-WINDOW:][None]).to(device)).cpu().numpy()[0]
pred_patch = pd.DataFrame(scaler_idx_tr.inverse_transform(raw), index=val.index, columns=INDEX_COLS)
rmse_patch = compute_rmse(val, pred_patch)
print(f"[PatchTST] RMSE = {rmse_patch:,.2f}")
per = np.sqrt(((val.values - pred_patch.values)**2).mean(axis=0))
for col, r in zip(INDEX_COLS, per): print(f"  {col}: {r:,.2f}")
"""

# ── Comparativa ────────────────────────────────────────────────────
C_CMP = """\
results = {"VanillaTransformer": rmse_vt, "TFT": rmse_tft, "PatchTST": rmse_patch}
df_res  = pd.DataFrame.from_dict(results, orient="index", columns=["RMSE_local"]).sort_values("RMSE_local")
print(df_res.round(2))

blended    = (pred_vt.values + pred_tft.values + pred_patch.values) / 3
pred_blend = pd.DataFrame(blended, index=val.index, columns=INDEX_COLS)
print(f"[Ensemble Transformers] RMSE = {compute_rmse(val, pred_blend):,.2f}")
"""

# ── Submission ─────────────────────────────────────────────────────
C_SUB = """\
scaled_full, scaler_full_idx = prepare_data(train_full, macro, network)
X_full, y_full = make_sequences(scaled_full, WINDOW, HORIZON)
Xf = torch.tensor(X_full); yf = torch.tensor(y_full)
n_feat_f = X_full.shape[2]
print(f"Full sequences: {X_full.shape}  n_feat={n_feat_f}")

def retrain(ModelClass):
    m   = ModelClass().to(device) if ModelClass is PatchTST else ModelClass(n_feat_f).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    train_loop(m, opt, Xf, yf)
    m.eval()
    with torch.no_grad():
        seed = torch.tensor(scaled_full[-WINDOW:][None]).to(device)
        raw  = m(seed).cpu().numpy()[0]
    return scaler_full_idx.inverse_transform(raw)

print("--- VanillaTransformer ---"); a_vt    = retrain(VanillaTransformer)
print("--- TFT ---");                a_tft   = retrain(TFT)
print("--- PatchTST ---");           a_patch = retrain(PatchTST)

arr_ens   = (a_vt + a_tft + a_patch) / 3
pred_test = pd.DataFrame(arr_ens, index=test_dates, columns=INDEX_COLS)
make_submission(pred_test, "submission_06_transformers.csv")
pred_test.head()
"""

transformer_nb = nb([
    md("# 06 - Transformers (PyTorch)\nVanilla Transformer, TFT, PatchTST.\n\nCPU: EPOCHS=10, D_MODEL=64. GPU: aumentar a EPOCHS=60, D_MODEL=128."),
    code(C_SETUP),
    code(C_DATA),
    md("## Positional Encoding (compartido)"),
    code(C_PE),
    md("---\n## Modelo 1 - Vanilla Transformer"),
    code(C_VT), code(C_VT_VAL),
    md("---\n## Modelo 2 - TFT (Temporal Fusion Transformer simplificado)"),
    code(C_TFT), code(C_TFT_VAL),
    md("---\n## Modelo 3 - PatchTST"),
    code(C_PATCH), code(C_PATCH_VAL),
    md("---\n## Comparativa y Ensemble"),
    code(C_CMP),
    md("## Submission — entrenar en datos completos"),
    code(C_SUB),
])
save(transformer_nb, "06_Transformers.ipynb")
print("Done.")
