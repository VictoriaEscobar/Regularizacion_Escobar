# Regularizacion_Escobar
Trabajo Práctico 3: Regularización - Redes Neuronales para Bioingeniería

Clasificación de documentos del dataset **20 Newsgroups** (20 clases) con un **Multilayer Perceptron(MLP)** en PyTorch, e incorporación y evaluación de técnicas de regularización (L1, L2, ElasticNet, Dropout, Batch Normalization, Max-Norm, Early Stopping).

El trabajo parte de un pipeline base (preprocesamiento de texto, TF-IDF y MLP con búsqueda de hiperparámetros por Optuna) y agrega una sección de regularización con experimentos controlados (una técnica por vez sobre un baseline fijo) y un cierre con Optuna que trata la regularización como hiperparámetro.


## Dataset

- **20 Newsgroups** (`sklearn.datasets.fetch_20newsgroups`), subsets `train` (11.314 docs) y `test` (7.532 docs).
- Se cargan con `remove=("headers", "footers", "quotes")` para evitar que el modelo aprenda metadatos en lugar de contenido.
- La descarga es automática en la primera ejecución (requiere conexión a internet).



## Estructura del notebook

| Sección | Contenido |
|---|---|
| 0 | Setup: instalación de dependencias y recursos de NLTK |
| 1–3 | Carga de datos, preprocesamiento configurable y vectorización TF-IDF |
| 4–5 | Modelo `MLPConfigurable` y función de entrenamiento |
| 6–7 | Búsqueda de hiperparámetros con Optuna (Softmax y MLP) y entrenamiento final |
| 8 | Evaluación final del modelo base (matriz de confusión, reporte por clase) |
| 9 | **Regularización**: clases/funciones `*_reg`, configuración común y experimentos controlados (Exp0 baseline, L2, L1 + poda, L1+L2, Dropout, Batch Normalization, "receta") |
| 9.10–9.11 | Comparación (tabla + gráficos) y evaluación en test del mejor modelo |
| 10 | Búsqueda con Optuna incluyendo la regularización como hiperparámetro |
| 11 | Análisis e interpretación de los resultados |


## Requisitos e instalación

- **Python 3.10–3.12** 
- Dependencias:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch scikit-learn pandas matplotlib tensorboard mlflow optuna nltk
```


## Cómo ejecutar

1. Abrir el notebook (`03_Regularizacion.ipynb`) en Jupyter, VS Code o Google Colab.
2. Seleccionar el kernel/intérprete del entorno donde se instalaron las dependencias.
3. **Ejecutar desde la celda 0** (*Run all*). La primera corrida descarga el dataset y los recursos de NLTK.

**Notas de ejecución:**
- Tiempo total aproximado: **~30–60 min en CPU** (lo más pesado es el preprocesamiento de las 6 combinaciones + las búsquedas de Optuna). En GPU la parte de entrenamiento es más rápida; el preprocesamiento de texto es CPU y no se acelera.
- Los estudios de Optuna usan `storage` SQLite (`optuna_*.db`): **re-ejecutar agrega más trials** al mismo estudio. Para empezar de cero, borrar esos archivos `.db`.
- Para una **primera corrida rápida**, bajar `n_trials` (celdas de Optuna del base) y `N_TRIALS_REG` (Sección 10) a ~5–8.
- El seguimiento se registra en MLflow (`mlruns/`) y TensorBoard (`runs/`).


## Metodología (resumen)

- Partición estricta **train / validación / test**: los hiperparámetros se eligen solo con validación y el **test se reserva** para la evaluación final.
- **Experimentos controlados y univariados**: un baseline fijo (TF-IDF 5000 features, MLP `[512, 256]`, ReLU, Glorot, Adam `lr=1e-3`) y se cambia **una sola técnica por vez** para poder atribuir el efecto observado.
- Semilla fija y `DataLoader` con generador propio para reproducibilidad.
- Ejes de análisis pedidos por la consigna: **magnitud de los pesos, tiempos de entrenamiento, learning rate y curvas de aprendizaje**.


## Resultados principales

- El **techo de validación** del problema (TF-IDF + MLP de dos capas) está en torno a 0,70; ninguna técnica lo supera de forma clara. La regularización sirve sobre todo para **controlar el sobreajuste**, no para subir el máximo.
- **L1/L2** achican la magnitud de los pesos (L2 de forma uniforme, L1 esparsa); **Dropout** regulariza sin tocar la magnitud; **Batch Normalization** acelera y estabiliza pero casi no regulariza en una red de dos capas.
- Mejor estrategia: **Dropout** (p ≈ 0,2–0,3), coincidente entre el análisis controlado y la búsqueda con Optuna.
- **Accuracy en test** del mejor modelo regularizado: ~0,63 (la caída val→test de ~6–7 puntos es consistente y propia de la partición de 20 Newsgroups).

El detalle completo está en la Sección 11 del notebook.

---

## Limitaciones

- Resultados con una sola semilla: diferencias de 1–2 puntos no son concluyentes.
- Los tiempos por época están dominados por el ruido de medición (CPU compartida); el costo real se refleja mejor en el número de épocas hasta converger.
- La restricción Max-Norm (c=4) no llegó a actuar (la norma máxima de fila nunca superó ese valor).

---

## Archivos

- `03_Regularizacion.ipynb`: notebook principal (ejecutado).
- `optuna_*.db`: estudios de Optuna persistidos (se generan al ejecutar).
- `mlruns/`, `runs/`: registros de MLflow y TensorBoard (se generan al ejecutar).
