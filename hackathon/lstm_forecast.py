"""
ENTREGA 4 — LSTM / Seq2Seq forecast.
Two modes:
  - "lstm":    Bidirectional LSTM, autorregressive step-by-step prediction.
  - "seq2seq": Encoder-Decoder that outputs all 252 steps in one pass.

Requires: torch (pip install torch --index-url https://download.pytorch.org/whl/cpu)
Falls back to a Keras/TensorFlow LSTM if torch is not available.
"""
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, ".")
from utils import (
    load_data, compute_rmse, make_submission, train_val_split,
    add_log_returns, INDEX_COLS,
)

warnings.filterwarnings("ignore")

WINDOW = 60       # look-back window (days)
HIDDEN = 128      # LSTM hidden size
N_LAYERS = 2
EPOCHS = 50
BATCH = 64
LR = 1e-3


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def build_sequences(values: np.ndarray, window: int, horizon: int = 1):
    """
    values: (T, F) array
    Returns X (N, window, F) and y (N, horizon, n_targets).
    """
    X, y = [], []
    n_targets = values.shape[1]  # first n_targets columns are the indices
    for i in range(window, len(values) - horizon + 1):
        X.append(values[i - window: i])
        y.append(values[i: i + horizon, :n_targets])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_data(train: pd.DataFrame, macro=None, network=None):
    """Scale and concatenate all features. Returns (scaled_array, scaler_indices, feature_df)."""
    df = train[INDEX_COLS].copy()

    if macro is not None:
        macro_aligned = macro.reindex(df.index).ffill().fillna(0)
        df = pd.concat([df, macro_aligned], axis=1)

    if network is not None:
        net_aligned = network.reindex(df.index).ffill().fillna(0)
        df = pd.concat([df, net_aligned], axis=1)

    df = df.ffill().fillna(0)

    # Separate scaler for the 6 index columns (needed for inverse transform)
    scaler_indices = StandardScaler()
    scaler_all = StandardScaler()

    scaled_indices = scaler_indices.fit_transform(df[INDEX_COLS].values)
    scaled_all = scaler_all.fit_transform(df.values)

    return scaled_all, scaler_indices, scaler_all, df.columns.tolist()


# ---------------------------------------------------------------------------
# PyTorch models
# ---------------------------------------------------------------------------

def _try_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


class BiLSTMModel:
    """Thin wrapper around a PyTorch bidirectional LSTM."""

    def __init__(self, input_size, hidden=HIDDEN, n_layers=N_LAYERS, n_out=6):
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden, n_layers,
                                    batch_first=True, bidirectional=True, dropout=0.2)
                self.fc = nn.Linear(hidden * 2, n_out)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        self.net = _Net()
        self.optim = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()
        self.torch = torch

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=EPOCHS, batch=BATCH):
        torch = self.torch
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y[:, 0, :], dtype=torch.float32)  # single step output
        n = len(X_t)
        self.net.train()
        for ep in range(epochs):
            idx = np.random.permutation(n)
            total_loss = 0.0
            for start in range(0, n, batch):
                b = idx[start: start + batch]
                xb, yb = X_t[b], y_t[b]
                self.optim.zero_grad()
                loss = self.loss_fn(self.net(xb), yb)
                loss.backward()
                self.optim.step()
                total_loss += loss.item() * len(b)
            if (ep + 1) % 10 == 0:
                print(f"    Epoch {ep+1}/{epochs}  loss={total_loss/n:.6f}")

    def predict_one(self, x: np.ndarray) -> np.ndarray:
        self.net.eval()
        torch = self.torch
        with torch.no_grad():
            xt = torch.tensor(x[None], dtype=torch.float32)
            return self.net(xt).numpy()[0]


class Seq2SeqModel:
    """Encoder-Decoder LSTM: encoder reads history, decoder outputs 252 steps."""

    def __init__(self, input_size, hidden=HIDDEN, n_out=6, horizon=252):
        import torch
        import torch.nn as nn

        class _Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden, N_LAYERS, batch_first=True, dropout=0.2)

            def forward(self, x):
                _, (h, c) = self.lstm(x)
                return h, c

        class _Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(n_out, hidden, N_LAYERS, batch_first=True, dropout=0.2)
                self.fc = nn.Linear(hidden, n_out)

            def forward(self, h, c, steps):
                # Start token: zeros
                batch = h.shape[1]
                dec_in = torch.zeros(batch, 1, n_out)
                outputs = []
                for _ in range(steps):
                    out, (h, c) = self.lstm(dec_in, (h, c))
                    pred = self.fc(out)
                    outputs.append(pred)
                    dec_in = pred
                return torch.cat(outputs, dim=1)  # (batch, steps, n_out)

        self.encoder = _Encoder()
        self.decoder = _Decoder()
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optim = torch.optim.Adam(params, lr=LR)
        self.loss_fn = nn.MSELoss()
        self.torch = torch
        self.horizon = horizon

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=EPOCHS, batch=BATCH):
        """y: (N, horizon, n_targets)."""
        torch = self.torch
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        n = len(X_t)
        self.encoder.train()
        self.decoder.train()
        for ep in range(epochs):
            idx = np.random.permutation(n)
            total_loss = 0.0
            for start in range(0, n, batch):
                b = idx[start: start + batch]
                xb, yb = X_t[b], y_t[b]
                self.optim.zero_grad()
                h, c = self.encoder(xb)
                out = self.decoder(h, c, yb.shape[1])
                loss = self.loss_fn(out, yb)
                loss.backward()
                self.optim.step()
                total_loss += loss.item() * len(b)
            if (ep + 1) % 10 == 0:
                print(f"    Epoch {ep+1}/{epochs}  loss={total_loss/n:.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.encoder.eval()
        self.decoder.eval()
        torch = self.torch
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32)
            h, c = self.encoder(Xt)
            out = self.decoder(h, c, self.horizon)
            return out.numpy()  # (N, horizon, n_out)


# ---------------------------------------------------------------------------
# Keras fallback
# ---------------------------------------------------------------------------

def fit_keras_lstm(X: np.ndarray, y: np.ndarray, epochs=EPOCHS):
    import tensorflow as tf
    from tensorflow import keras

    input_size = X.shape[2]
    n_out = y.shape[2]

    inp = keras.Input(shape=(WINDOW, input_size))
    x = keras.layers.Bidirectional(keras.layers.LSTM(HIDDEN, return_sequences=False))(inp)
    x = keras.layers.Dense(64, activation="relu")(x)
    out = keras.layers.Dense(n_out)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y[:, 0, :], epochs=epochs, batch_size=BATCH, verbose=1)
    return model


# ---------------------------------------------------------------------------
# Training and inference wrappers
# ---------------------------------------------------------------------------

def train_and_predict(
    scaled_train: np.ndarray,
    scaler_indices: StandardScaler,
    test_dates: pd.DatetimeIndex,
    mode: str = "seq2seq",
) -> np.ndarray:
    n_features = scaled_train.shape[1]
    n_out = 6
    n_steps = len(test_dates)  # dinámico: funciona con cualquier horizonte de test

    if mode == "seq2seq":
        X, y = build_sequences(scaled_train, WINDOW, horizon=n_steps)
        torch = _try_torch()
        if torch:
            print(f"  Building Seq2Seq (PyTorch) on {X.shape[0]} samples, horizon={n_steps} ...")
            model = Seq2SeqModel(n_features, hidden=HIDDEN, n_out=n_out, horizon=n_steps)
            model.fit(X, y, epochs=EPOCHS)
            seed = scaled_train[-WINDOW:][None]
            raw = model.predict(seed)[0]  # (n_steps, 6)
        else:
            print("  PyTorch not found; falling back to keras BiLSTM (step-by-step).")
            mode = "lstm"

    if mode == "lstm":
        X, y = build_sequences(scaled_train, WINDOW, horizon=1)
        torch = _try_torch()
        if torch:
            print(f"  Building BiLSTM (PyTorch) on {X.shape[0]} samples, horizon={n_steps} ...")
            model = BiLSTMModel(n_features, hidden=HIDDEN, n_out=n_out)
            model.fit(X, y, epochs=EPOCHS)
            window = scaled_train[-WINDOW:].copy()
            raw = []
            for _ in range(n_steps):
                pred = model.predict_one(window)
                raw.append(pred)
                new_row = window[-1].copy()
                new_row[:6] = pred
                window = np.vstack([window[1:], new_row])
            raw = np.array(raw)
        else:
            try:
                print("  Building BiLSTM (Keras) ...")
                m = fit_keras_lstm(X, y, epochs=EPOCHS)
                window = scaled_train[-WINDOW:].copy()
                raw = []
                for _ in range(n_steps):
                    pred = m.predict(window[None], verbose=0)[0]
                    raw.append(pred)
                    new_row = window[-1].copy()
                    new_row[:6] = pred
                    window = np.vstack([window[1:], new_row])
                raw = np.array(raw)
            except ImportError:
                raise RuntimeError("Neither PyTorch nor TensorFlow found. Install one of them.")

    # Inverse-transform the 6 index columns
    preds = scaler_indices.inverse_transform(raw)
    return preds


# ---------------------------------------------------------------------------
# Validation + main
# ---------------------------------------------------------------------------

def local_validate(data: dict, mode: str = "seq2seq") -> float:
    train_full = data["train_indices"][INDEX_COLS]
    train, val = train_val_split(train_full, val_size=252)

    macro = data.get("train_macro_factors")
    network = data.get("train_network_metrics")

    macro_tr = macro.iloc[:-252] if macro is not None else None
    net_tr = network.iloc[:-252] if network is not None else None

    scaled_train, scaler_idx, _, _ = prepare_data(train, macro_tr, net_tr)

    print(f"Running {mode} validation ...")
    preds_arr = train_and_predict(scaled_train, scaler_idx, val.index, mode=mode)
    pred_df = pd.DataFrame(preds_arr, index=val.index, columns=INDEX_COLS)

    rmse = compute_rmse(val, pred_df)
    print(f"[LSTM/{mode}] Local validation RMSE: {rmse:.2f}")
    per_index = np.sqrt(((val.values - pred_df.values) ** 2).mean(axis=0))
    for col, r in zip(INDEX_COLS, per_index):
        print(f"  {col}: {r:.2f}")
    return rmse


def main(mode: str = "seq2seq"):
    data = load_data()
    train = data["train_indices"][INDEX_COLS]
    test_dates = data["test_dates"].index

    macro = data.get("train_macro_factors")
    network = data.get("train_network_metrics")

    local_validate(data, mode=mode)

    print("\nTraining on full dataset ...")
    scaled_train, scaler_idx, _, _ = prepare_data(train, macro, network)
    preds_arr = train_and_predict(scaled_train, scaler_idx, test_dates, mode=mode)
    pred_df = pd.DataFrame(preds_arr, index=test_dates, columns=INDEX_COLS)

    make_submission(pred_df, f"submission_04_lstm_{mode}.csv")
    print("Done.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["lstm", "seq2seq"], default="seq2seq")
    args = p.parse_args()
    main(mode=args.mode)
