# MIAX14 Hackathon - Predicción de Índices Bursátiles

**Objetivo**: Predecir 6 índices financieros sintéticos para 173 días de trading (2028-12-13 → 2029-08-21).  
**Métrica**: RMSE promedio entre todos los índices. Mínimo aceptable: RMSE < 75 000.  
**Entregas**: Máximo 6 en la plataforma.

## Índices

| Índice | Descripción |
|--------|-------------|
| Index_A | Alpha-Tech – alta volatilidad, alto crecimiento |
| Index_B | Steady-State – baja volatilidad, defensivo |
| Index_C | Energy-Pulse – energía global y macroeconomía |
| Index_D | The Ghost – sigue una señal oculta en otro índice |
| Index_E | Global-ESG – mezcla de empresas sostenibles |
| Index_F | Digital-Frontier – volatilidad extrema, mercados digitales (crypto) |

## Estructura del repositorio

```
hackathon_miax14/
├── data/                        # Datos del hackathon (ignorados por git salvo csv)
│   ├── train_indices.csv
│   ├── train_macro_factors.csv
│   ├── train_network_metrics.csv
│   ├── train_news.csv
│   ├── test_macro_factors.csv
│   ├── test_network_metrics.csv
│   └── test_news.csv
├── notebooks/
│   ├── 01_eda.ipynb             # Exploración completa de los datos
│   ├── 02_modeling.ipynb        # Entrenamiento, validación, submission
│   └── 03_index_d_detective.ipynb  # Análisis de la señal oculta de Index_D
├── src/
│   ├── features.py              # Feature engineering
│   ├── models.py                # LightGBM multi-índice
│   ├── predict.py               # Predicción autorregresiva
│   └── train.py                 # Script principal de entrenamiento
├── submissions/                 # Archivos xlsx para subir
├── requirements.txt
└── README.md
```

## Quick start

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar entrenamiento y generar submission
cd src
python train.py --data-dir ../data --output ../submissions/submission_v1.xlsx

# 3. O usar los notebooks (recomendado)
jupyter lab
```

## Pipeline de modelado

1. **Features**: lags (1,2,3,5,10,21,63 días), rolling mean/std (5,10,21,63), retornos (1,5,21 días), factores macro, métricas de red para Index_F, conteo de noticias, variables de calendario.

2. **Modelo**: LightGBM independiente por índice con early stopping. Un modelo por índice permite capturar dinámicas distintas.

3. **Predicción autorregresiva**: para cada día de test, se construye el vector de features usando la historia conocida (train + predicciones anteriores) y se predice. La predicción se añade al histórico antes del siguiente paso.

4. **Index_D**: se investiga en `03_index_d_detective.ipynb`. Sigue a `Index_A` con desfase 1: `Index_D(t) ≈ 1.0001815 · Index_A(t-1) + 3.116` (R²=0.99999).

5. **Index_F** (Digital-Frontier): plano en 1000 hasta 2020-03-10 (el activo "no existía"). Se entrena solo con datos post-2020 para evitar el sesgo del valor plano.

## Sistema alcista para índices tecnológicos (Index_A, Index_D)

`Index_A` y `Index_D` son tecnológicas con tendencia alcista fuerte (~16% CAGR). Los modelos de árboles **no extrapolan**: en la submission v1, `Index_A` colapsó **-29.5%** en el horizonte autorregresivo. El sistema (`src/bullish_guard.py`) lo corrige en dos partes:

1. **Reentrenamiento con todos los datos** antes de generar el submission. El modelo se valida con holdout, pero el modelo final se reentrena con la serie completa hasta la última fecha. Solo esto pasó `Index_A` de -29.5% a +2.8%.

2. **Guard-rail de drift**: se mezcla la predicción del modelo con la proyección exponencial del drift histórico (`blend_with_drift`) y se acota dentro de una banda alrededor del drift (`apply_drift_guard`), evitando tanto el colapso como la explosión.

Resultado: `Index_A` pasa de **-29.5% → +3.2%** y `Index_D` de **-28.4% → +4.3%** (dirección alcista correcta).

## Red neuronal + network anchor (Index_A, Index_B, Index_D)

Notebook `05_neural_network.ipynb` y módulos `src/neural_model.py`, `src/network_anchor.py`.

- **MLP sobre log-retornos** (`NeuralReturnModel`): a diferencia de los árboles, una red entrenada para
  predecir el log-retorno diario (target estacionario) puede seguir subiendo más allá del máximo
  histórico, reconstruyendo el nivel por composición `nivel(t)=nivel(t-1)·exp(r̂)`. Usa solo features
  **estacionarias** (retornos + exógenas) y clip de ±10%/día para no explotar en autorregresivo.
  En validación gana a los árboles en A, B y D (A: 161k→101k RMSE).
- **Network anchor** (`network_anchor.py`): las network metrics correlacionan **~0.90 con Index_A en
  niveles** (no en retornos) y sus valores en el test son **futuro conocido**. Blend 80% NN + 20% anchor
  baja el RMSE de A de **101k → 91k**. Como las network suben +90% en el test, anclan A al alza.
- **Index_D = ghost(A)**: se deriva de la A ya corregida, manteniendo D≈A.

## Post-proceso con feedback real del leaderboard (submission 5)

`src/postprocess.py` + `src/build_submission5.py`. Tras subir la submission 3 (RMSE global 79,405):

| Índice | RMSE real | Predicción |
|--------|----------:|-----------|
| Index_A | 232,402 | +34% (se pasó) |
| Index_D | 152,874 | +28.6% (mejor) |
| Index_B | 70,106 | −14.5% (colapso) |

- **A sigue a D**: `D(t) ≈ A(t-1)` con corr **0.999997** (misma serie, lag 1). La predicción de D
  (+28.6%) acertó más que la de A (+34%), así que reconstruimos A para que siga la trayectoria de D:
  `A(t) = A_last · D_pred(t)/D_last`. A y D quedan idénticos (diferencia 0.000%).
- **Index_B forzado** a la red neuronal de una compañera (`results/locked_predictions_B.csv`,
  exógenas + macro + network), más realista (−2%, oscilante) que nuestro colapso a −14.5%.

## ¿Predice retornos o precios?

Depende del componente: la **red neuronal** (`neural_model.py`) predice **log-retornos** (target
estacionario) con cap de ±10%/día y reconstruye el nivel por composición; los **árboles** predicen
niveles directamente; A y D en la submission final siguen la trayectoria de D (post-proceso).

## Estrategia de mejora iterativa

| Intento | Estrategia |
|---------|-----------|
| 1 | Baseline LightGBM con lags simples |
| 2 | Explotar señal oculta de Index_D |
| 3 | Tuning de hiperparámetros con Optuna |
| 4 | Ensemble LightGBM + XGBoost |
| 5 | Incorporar análisis de sentimiento de noticias |
| 6 | Modelo final optimizado |

## Validación

Se usa los últimos 252 días de training como conjunto de validación (out-of-sample), simulando el horizonte de predicción real.
