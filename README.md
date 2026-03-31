# CorAI — Clasificador de Ritmos Cardiacos

Sistema de clasificacion automatica de ritmos cardiacos basado en aprendizaje automatico,
entrenado con la base de datos AFDB de PhysioNet y senales propias del dispositivo CorAI.

---

## Tabla de contenidos

1. [Clases del modelo](#1-clases-del-modelo)
2. [Archivos generados por el entrenamiento](#2-archivos-generados-por-el-entrenamiento)
3. [Como usar los artefactos en una API](#3-como-usar-los-artefactos-en-una-api)
4. [Ejemplo completo con FastAPI](#4-ejemplo-completo-con-fastapi)
5. [Ejemplo completo con Flask](#5-ejemplo-completo-con-flask)
6. [Formato de entrada esperado](#6-formato-de-entrada-esperado)
7. [Formato de respuesta](#7-formato-de-respuesta)
8. [Integracion con el frontend](#8-integracion-con-el-frontend)
9. [Rendimiento del modelo](#9-rendimiento-del-modelo)
10. [Notas importantes](#10-notas-importantes)

---

## 1. Clases del modelo

El modelo distingue cuatro tipos de ritmo cardiaco:

Atrial 1 = AFIB (Fibrilación Auricular) y Atrial 2 = AFL (Flutter Auricular)

| Etiqueta | Nombre clinico              | Descripcion breve                                      |
|----------|-----------------------------|--------------------------------------------------------|
| `AFIB`   | Fibrilacion Auricular        | Ritmo irregularmente irregular, sin onda P definida    |
| `AFL`    | Flutter Auricular            | Ritmo regular con ondas F en dientes de sierra         |
| `J`      | Ritmo Juncional AV           | Ritmo regular originado en el nodo AV                  |
| `N`      | Ritmo sinusal normal         | Ritmo normal de referencia                             |

---

## 2. Archivos generados por el entrenamiento

Despues de correr `afdb_supervised_classification.py --save-model`, el directorio
`results_supervised/` (o el que se haya indicado en `--results-dir`) contiene:

### Artefactos del modelo (necesarios para inferencia)

```
results_supervised/
├── afdb_rhythm_classifier.joblib   # Clasificador RandomForest entrenado
├── afdb_scaler.joblib              # StandardScaler ajustado sobre el training set
├── afdb_label_encoder.joblib       # LabelEncoder: convierte numeros a etiquetas (AFIB, AFL, J, N)
├── afdb_feature_columns.joblib     # Lista ordenada de los 39 nombres de features
└── afdb_col_medians.joblib         # Medianas de cada feature para imputar NaN
```

> **Estos cinco archivos deben mantenerse juntos y versionados en conjunto.**
> Usar el scaler de una version diferente al clasificador produce resultados incorrectos.

### Archivos de evaluacion y reporte (solo referencia, no necesarios para inferencia)

```
results_retrained/
├── report.txt                  # Reporte narrativo legible, sobreescrito en cada run
├── metrics_<timestamp>.json    # Metricas completas en JSON, acumuladas por run
├── evaluation_summary.csv      # Tabla consolidada de accuracy por fuente y tipo de ruido
├── predictions.csv             # Predicciones por senal con columna source
├── confusion_matrix.png        # Matriz de confusion del split interno AFDB
├── feature_importance.png      # Top 20 features mas importantes
├── pca_visualization.png       # PCA del espacio de features de entrenamiento
└── class_distribution.png      # Distribucion de clases en el training set
```

---

## 3. Como usar los artefactos en una API

### Instalacion de dependencias

```bash
pip install fastapi uvicorn flask joblib scikit-learn numpy scipy pywt
```

### Carga de artefactos (compartida entre Flask y FastAPI)

Crea un archivo `model.py` en tu proyecto de API:

```python
# model.py
import joblib
import numpy as np
from scipy.signal import butter, filtfilt
import pywt

# --- Carga de artefactos ---
# Ajusta la ruta al directorio donde guardaste los .joblib
MODEL_DIR = "results_retrained"

classifier    = joblib.load(f"{MODEL_DIR}/afdb_rhythm_classifier.joblib")
scaler        = joblib.load(f"{MODEL_DIR}/afdb_scaler.joblib")
label_encoder = joblib.load(f"{MODEL_DIR}/afdb_label_encoder.joblib")
feature_cols  = joblib.load(f"{MODEL_DIR}/afdb_feature_columns.joblib")
col_medians   = joblib.load(f"{MODEL_DIR}/afdb_col_medians.joblib")

# Frecuencia de muestreo del dispositivo CorAI
DEVICE_FS = 200.0


def preprocess_signal(signal: list[float]) -> np.ndarray:
    """
    Convierte una lista de muestras v_raw a un array float32 y aplica
    el filtro pasa-banda 0.5-40 Hz para atenuar ruido muscular y deriva
    de linea base antes de extraer features.
    """
    sig = np.array(signal, dtype=np.float32)
    nyq = 0.5 * DEVICE_FS
    b, a = butter(4, [0.5 / nyq, 40.0 / nyq], btype="band")
    return filtfilt(b, a, sig).astype(np.float32)


def extract_features(signal: np.ndarray) -> dict:
    """
    Extrae el vector de 39 features identico al usado durante el entrenamiento.
    Importa extract_all_features desde afdb_dataset_loader.py del proyecto
    de entrenamiento (o copia la funcion aqui si la API es independiente).
    """
    from afdb_dataset_loader import extract_all_features
    return extract_all_features(signal, DEVICE_FS)


def predict(signal: list[float]) -> dict:
    """
    Recibe una lista de muestras v_raw, ejecuta el pipeline completo y
    devuelve la clase predicha junto con las probabilidades por clase.

    Parametros
    ----------
    signal : list of float
        Muestras de voltaje crudo (v_raw) del dispositivo.

    Retorna
    -------
    dict con keys: predicted_class, probabilities, n_samples
    """
    # 1. Preprocesamiento
    sig_filtered = preprocess_signal(signal)

    # 2. Extraccion de features
    feats_dict = extract_features(sig_filtered)

    # 3. Construir vector en el orden exacto que espera el scaler
    feature_vector = np.array(
        [feats_dict.get(col, np.nan) for col in feature_cols],
        dtype=np.float64,
    ).reshape(1, -1)

    # 4. Imputar NaN con medianas del training set
    nan_mask = np.isnan(feature_vector)
    feature_vector[nan_mask] = col_medians[np.where(nan_mask)[1]]

    # 5. Normalizar y clasificar
    X_scaled = scaler.transform(feature_vector)
    pred_enc  = classifier.predict(X_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_enc])[0]

    # 6. Probabilidades por clase
    proba = classifier.predict_proba(X_scaled)[0]
    probabilities = {
        cls: round(float(p), 4)
        for cls, p in zip(label_encoder.classes_, proba)
    }

    return {
        "predicted_class": pred_label,
        "probabilities":   probabilities,
        "n_samples":       int(len(signal)),
    }
```

---

## 4. Ejemplo completo con FastAPI

```python
# api_fastapi.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from model import predict

app = FastAPI(
    title="CorAI ECG Classifier",
    description="Clasificacion automatica de ritmos cardiacos a partir de senales ECG.",
    version="1.0.0",
)


class ECGRequest(BaseModel):
    samples: list[float] = Field(
        ...,
        description="Lista de muestras v_raw del dispositivo CorAI (minimo 500 muestras).",
        min_length=500,
    )


class PredictionResponse(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]
    n_samples: int


@app.post("/predict", response_model=PredictionResponse)
def classify_ecg(request: ECGRequest):
    """
    Recibe una senal ECG como lista de muestras v_raw y devuelve
    la clase de ritmo predicha junto con las probabilidades por clase.
    """
    try:
        result = predict(request.samples)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    """Verifica que la API y el modelo estan cargados correctamente."""
    return {"status": "ok", "model": "afdb_rhythm_classifier"}
```

**Ejecutar:**

```bash
uvicorn api_fastapi:app --host 0.0.0.0 --port 8000 --reload
```

**Documentacion interactiva disponible en:** `http://localhost:8000/docs`

---

## 5. Ejemplo completo con Flask

```python
# api_flask.py
from flask import Flask, request, jsonify
from model import predict

app = Flask(__name__)


@app.post("/predict")
def classify_ecg():
    """
    Recibe un JSON con la lista de muestras v_raw y devuelve
    la clase predicha con probabilidades.

    Body esperado:
        { "samples": [1.26, 1.23, 1.27, ...] }
    """
    data = request.get_json(force=True)

    if not data or "samples" not in data:
        return jsonify({"error": "Se requiere el campo 'samples'."}), 400

    samples = data["samples"]

    if len(samples) < 500:
        return jsonify({"error": "Se requieren al menos 500 muestras."}), 400

    try:
        result = predict(samples)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": "afdb_rhythm_classifier"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
```

**Ejecutar:**

```bash
python api_flask.py
```

---

## 6. Formato de entrada esperado

La API acepta la senal ECG directamente como un array de muestras `v_raw`.

### Opcion A — Array de muestras directo (recomendado)

```json
{
  "samples": [1.266598, 1.232683, 1.276966, 1.251234, ...]
}
```

### Opcion B — Si el frontend envia el JSON completo del dispositivo

Extrae el campo `v_raw` antes de enviarlo a la API:

```javascript
// JavaScript / frontend
const ecgJson = await fetch("archivo.json").then(r => r.json());
const samples = ecgJson.ecg.map(s => s.v_raw);

const response = await fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ samples }),
});

const result = await response.json();
console.log(result.predicted_class);   // "AFIB", "AFL", "J" o "N"
console.log(result.probabilities);     // { AFIB: 0.87, AFL: 0.05, J: 0.01, N: 0.07 }
```

### Consideraciones de la senal de entrada

| Parametro            | Valor requerido                  |
|----------------------|----------------------------------|
| Frecuencia muestreo  | 200 Hz                           |
| Longitud minima      | 500 muestras (~2.5 s)            |
| Longitud recomendada | 12,000 muestras (60 s, un bloque de entrenamiento) |
| Campo a usar         | `v_raw` del JSON del dispositivo |
| Tipo de dato         | `float` (voltaje crudo)          |

> **Nota:** El modelo fue entrenado con bloques de 60 segundos (12,000 muestras a 200 Hz).
> Senales mas cortas funcionan pero pueden reducir la precision de los features HRV
> si no contienen suficientes latidos para calcular intervalos RR.

---

## 7. Formato de respuesta

```json
{
  "predicted_class": "AFIB",
  "probabilities": {
    "AFIB": 0.8721,
    "AFL":  0.0513,
    "J":    0.0089,
    "N":    0.0677
  },
  "n_samples": 12000
}
```

| Campo             | Tipo     | Descripcion                                              |
|-------------------|----------|----------------------------------------------------------|
| `predicted_class` | `string` | Clase con mayor probabilidad: `AFIB`, `AFL`, `J` o `N`  |
| `probabilities`   | `object` | Probabilidad de cada clase (suma = 1.0)                  |
| `n_samples`       | `int`    | Numero de muestras recibidas                             |

---

## 8. Integracion con el frontend

El frontend solo necesita hacer un `POST /predict` con el array de muestras.
La respuesta incluye tanto la clase predicha como las probabilidades, lo que
permite mostrar tanto el diagnostico como una barra de confianza por clase.

### Ejemplo de visualizacion sugerida

```
Resultado:  AFIB — Fibrilacion Auricular
Confianza:  87.2%

Probabilidades:
  AFIB  ████████████████████░░░  87.2%
  N     ██░░░░░░░░░░░░░░░░░░░░░   6.8%
  AFL   █░░░░░░░░░░░░░░░░░░░░░░   5.1%
  J     ░░░░░░░░░░░░░░░░░░░░░░░   0.9%
```

### Mapeo de etiquetas para mostrar al usuario

```javascript
const LABELS = {
  AFIB: "Fibrilacion Auricular",
  AFL:  "Flutter Auricular",
  J:    "Ritmo Juncional AV",
  N:    "Ritmo Sinusal Normal",
};

const COLORS = {
  AFIB: "#E74C3C",  // rojo — arritmia significativa
  AFL:  "#E67E22",  // naranja — arritmia moderada
  J:    "#F1C40F",  // amarillo — arritmia leve
  N:    "#2ECC71",  // verde — normal
};
```

---

## 9. Rendimiento del modelo

Resultados sobre el conjunto de prueba del ultimo entrenamiento:

| Fuente              | Condicion    | Accuracy |
|---------------------|--------------|----------|
| AFDB (split interno)| test_split   | 97.39%   |
| Dispositivo CorAI   | Limpia       | 100%     |
| Dispositivo CorAI   | Muscular     | 66.67%   |
| Dispositivo CorAI   | Respiracion  | 100%     |
| Augmentada (250 Hz) | Limpia       | 100%     |
| Augmentada (250 Hz) | Muscular     | 100%     |

> El rendimiento sobre senales con ruido muscular real (66.67%) esta basado en
> solo 9 muestras. Se recomienda ampliar el conjunto de prueba para una estimacion
> estadisticamente robusta.

---

## 10. Notas importantes

**Los cinco artefactos son inseparables.**
`classifier`, `scaler`, `label_encoder`, `feature_columns` y `col_medians` deben
corresponder al mismo run de entrenamiento. Si se reentrena el modelo, todos deben
actualizarse simultaneamente.

**El orden de las features es critico.**
El vector de entrada al scaler debe construirse en el orden exacto definido en
`afdb_feature_columns.joblib`. No construyas el vector manualmente; usa siempre
`feature_cols` para garantizarlo.

**La imputacion de NaN es obligatoria.**
Senales cortas o con deteccion fallida de picos R producen `NaN` en los features
HRV y QRS. Estos deben reemplazarse con `col_medians` antes de llamar a
`scaler.transform()`. Pasar `NaN` al scaler provoca errores o predicciones silenciosas
incorrectas.

**Frecuencia de muestreo fija.**
El pipeline esta calibrado para 200 Hz. Senales capturadas a otra frecuencia
requieren resampleo previo con `afdb_augment.py` (opcion `--src-fs` / `--dst-fs`).

**No es un dispositivo medico certificado.**
Las predicciones del modelo son orientativas y no deben usarse como unico
criterio de diagnostico clinico.