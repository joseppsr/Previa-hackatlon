# Hackathon MIAX14 — Guía de ejecución

## Setup (una vez)

```bash
pip install pandas numpy scikit-learn matplotlib statsmodels lightgbm pmdarima
pip install torch --index-url https://download.pytorch.org/whl/cpu   # PyTorch CPU
pip install transformers accelerate                                    # HuggingFace
pip install git+https://github.com/amazon-science/chronos-forecasting.git  # Chronos
pip install uni2ts                                                     # Moirai (opcional)
```

## Estructura de datos esperada

Coloca los CSVs del hackathon en una carpeta `data/` al mismo nivel que `hackathon/`:

```
previa hackatlon/
├── data/
│   ├── train_indices.csv
│   ├── test_dates.csv
│   ├── train_macro_factors.csv
│   ├── test_macro_factors.csv
│   ├── train_network_metrics.csv
│   ├── test_network_metrics.csv
│   ├── train_news.csv
│   └── test_news.csv
└── hackathon/
    ├── utils.py
    ├── baseline.py
    ├── arima_models.py
    ├── lgbm_forecast.py
    ├── lstm_forecast.py
    ├── ensemble.py
    ├── 00_eda.py
    └── submissions/
```

## Orden de ejecución (desde la carpeta `hackathon/`)

### 0. EDA (siempre primero — 10 min)
```bash
python 00_eda.py
```
Genera `eda_indices.png`, `eda_correlation.png`, y detecta el índice fuente de Index_D.

### Entrega 1 — Baseline (~15 min)
```bash
python baseline.py
# → submissions/submission_01_baseline.csv
```

### Entrega 2 — ARIMA (~30-60 min)
```bash
python arima_models.py
# → submissions/submission_02_arima.csv
```

### Entrega 3 — LightGBM (~30 min)
```bash
python lgbm_forecast.py
# → submissions/submission_03_lgbm.csv
```

### Entrega 4 — LSTM/Seq2Seq (~60-90 min con CPU)
```bash
python lstm_forecast.py --mode seq2seq   # recomendado
python lstm_forecast.py --mode lstm      # alternativa
# → submissions/submission_04_lstm_seq2seq.csv
```

### Entrega 5 — Ensemble + Ghost correction (~30 min)
```bash
python ensemble.py
# → submissions/submission_05_ensemble.csv
```

### Entrega extra A — Transformers propios (PyTorch)
Notebook: `06_Transformers.ipynb`
Tres arquitecturas implementadas desde cero:
- **Vanilla Transformer** (encoder-only, decoder lineal directo)
- **TFT** — Temporal Fusion Transformer con Variable Selection Network
- **PatchTST** — series divididas en patches, channel-independent

```
# → submissions/submission_06_transformers.csv
```

### Entrega extra B — Modelos Hugging Face
Notebook: `07_HuggingFace.ipynb`
Tres modelos de la librería `transformers` y fundations models:
- **TimeSeriesTransformer** (HF `transformers`) — fine-tuneable, distribución student-t
- **Chronos-T5-Small** (Amazon) — zero-shot, no requiere entrenamiento
- **Moirai** (Salesforce) — zero-shot multivariante

Al final del notebook genera también un **Mega-Ensemble** mezclando todos los submissions disponibles.

```
# → submissions/submission_07a_hf_tst.csv
# → submissions/submission_07b_chronos.csv
# → submissions/submission_07c_moirai.csv
# → submissions/submission_08_mega_ensemble.csv
```

## Notebooks disponibles

| Notebook | Contenido | Entrega |
|----------|-----------|---------|
| `00_EDA.ipynb` | Visualización, correlaciones, ghost detection | — |
| `01_Baseline.ipynb` | Naive / rolling / exp. smoothing | 1ª |
| `02_ARIMA.ipynb` | ARIMA auto-order | 2ª |
| `03_LightGBM.ipynb` | LightGBM + lag features + autoregresión | 3ª |
| `04_LSTM.ipynb` | Seq2Seq PyTorch (encoder→decoder 252 pasos) | 4ª |
| `05_Ensemble.ipynb` | Blend ponderado + corrección Index_D | 5ª |
| `06_Transformers.ipynb` | VanillaTransformer, TFT, PatchTST | extra |
| `07_HuggingFace.ipynb` | HF TST, Chronos, Moirai + Mega-Ensemble | extra |

## Consejo de uso de las 6 entregas del hackathon

| # | Cuándo | Notebook |
|---|--------|----------|
| 1 | Primeros 20 min | 01_Baseline.ipynb |
| 2 | ~1h | 02_ARIMA.ipynb |
| 3 | ~2h | 03_LightGBM.ipynb |
| 4 | ~3h | 04_LSTM.ipynb |
| 5 | ~4h | 05_Ensemble.ipynb |
| 6 | ~5h | 06_Transformers.ipynb o 07_HuggingFace.ipynb |

## Validación local antes de enviar

Cada script imprime el RMSE local (sobre los últimos 252 días de train).  
**Regla:** solo enviar si RMSE local < 75.000.
