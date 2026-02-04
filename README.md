# Previsión de demanda con Machine Learning (LightGBM)

Este proyecto desarrolla un **modelo de previsión de demanda** utilizando técnicas de **machine learning para series temporales**, con el objetivo de predecir ventas futuras a nivel **tienda–producto–fecha**.

El enfoque combina **ingeniería de características basada en el tiempo**, validación temporal y un modelo **LightGBM**, optimizado con una **función de costo personalizada (SMAPE)**.

---

## 📦 Contexto del problema

La previsión de demanda es un componente clave en:
- gestión de inventarios
- planificación de la cadena de suministro
- reducción de quiebres de stock y sobrecostos

Este proyecto aborda el problema desde un enfoque **data-driven**, utilizando históricos de ventas para anticipar la demanda futura.

---

## 🎯 Objetivo de Machine Learning

- **Tipo de problema:** Regresión (series temporales)
- **Variable objetivo:** Ventas
- **Horizonte de predicción:** múltiples períodos futuros
- **Métrica de evaluación:** SMAPE (Symmetric Mean Absolute Percentage Error)

---

## 📊 Dataset

- Datos históricos de ventas por:
  - tienda (`store`)
  - producto (`item`)
  - fecha (`date`)
- Separación explícita de conjuntos:
  - entrenamiento
  - validación
  - test

---

## 🧪 Metodología

### 1. Análisis exploratorio (EDA)
- Revisión de tipos de datos
- Análisis de valores faltantes
- Estadísticas descriptivas
- Comportamiento de ventas por tienda y producto

### 2. Feature Engineering (clave del proyecto)
- **Características temporales**
  - mes, día, semana, año
  - fines de semana, inicio/fin de mes
- **Lags de ventas**
  - 91 a 728 días
- **Rolling means**
  - ventanas anuales y semianuales
- **Exponentially Weighted Means (EWM)**
- **Codificación one-hot**
  - tienda, producto, día de la semana, mes
- **Transformación logarítmica**
  - `log1p(sales)`

---

## 🧠 Modelo

- **Algoritmo:** LightGBM Regressor
- **Validación:** split temporal (no aleatorio)
- **Early stopping**
- **Función de evaluación personalizada**
  - SMAPE implementada desde cero


SMAPE = (|y_pred - y_true| / (|y_pred| + |y_true|)) * 200
📈 Evaluación
- Evaluación en conjunto de validación temporal
- Optimización de hiperparámetros
- Selección de variables según:
  - Importancia por ganancia (gain)
  - Eliminación de features sin aporte

🏆 Resultados
- Modelo final entrenado con todos los datos históricos
- Predicciones generadas para el conjunto de test
- Archivo final de salida: submission_demand.csv

🛠️ Tecnologías utilizadas
- Python
- pandas, numpy
- matplotlib, `seaborn`
- LightGBM
- scikit-learn
- `missingno`

📂 Estructura del repositorio
├── demanda/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── Previsión de la demanda.py
├── submission_demand.csv
├── README.md

🚀 Próximos pasos
- Backtesting con ventanas móviles
- Comparación con modelos clásicos (ARIMA / SARIMA)
- Incorporación de variables externas (promociones, eventos)
- Deploy del modelo como servicio de forecasting
- Automatización del pipeline (MLflow / Airflow)

👤 Autor

Flavia Hepp
Data Scientist en formación
