# Predicción de Cancelación de Tiquetes en Agencias de Transporte

Modelo de aprendizaje **supervisado** que predice si un pasajero cancelará una reserva de transporte (bus, tren o avión) a partir de las características de la compra, para que la agencia pueda anticiparse y reducir pérdidas por cancelaciones y no-shows.

## Contexto del problema

Las agencias de transporte pierden dinero cuando un pasajero cancela a último momento o no se presenta. Usando datos reales de reservas (dataset de una competencia de Kaggle), este proyecto construye un clasificador que estima el riesgo de cancelación de cada reserva, permitiendo acciones preventivas como ajuste de precios, overbooking controlado o alertas tempranas.

- **Dataset:** `train.csv` — reservas individuales de bus, tren y avión.
- **Variable objetivo:** `Cancelled` (1 = canceló, 0 = se presentó).
- **Features principales:** días de anticipación de la compra, precio, descuentos, género, vuelo nacional/internacional, medio de transporte, motivo del viaje y estado interno de la reserva.

## Qué se usó

- **Lenguaje y entorno:** Python (Jupyter Notebook).
- **Librerías:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.
- **Modelo:** `RandomForestClassifier` con `class_weight="balanced"` para compensar el desbalance (~15% de cancelaciones).
- **Pipeline:**
  1. Limpieza de datos y cálculo de `dias_anticipacion` a partir de fechas de compra y salida.
  2. Codificación de variables categóricas con `LabelEncoder`.
  3. División entrenamiento/prueba (80/20) estratificada.
  4. Entrenamiento y análisis de importancia de variables.
- **Evaluación:** accuracy, validación cruzada (5 folds), curva ROC / AUC y matriz de confusión.

## Qué se logró

- Un clasificador capaz de distinguir reservas con alto y bajo riesgo de cancelación, validado con validación cruzada y curva ROC/AUC.
- **Hallazgo clave:** el estado interno de la reserva (`StatusofReserve`) es, de lejos, la variable más determinante (~76% de importancia): ciertos estados nunca se cancelan y otros siempre lo hacen. El precio (~9%) y los días de anticipación aportan en menor medida.
- Análisis exploratorio que muestra cómo varía la tasa de cancelación según el medio de transporte, el precio y la anticipación de la compra.
- Una función de predicción sobre reservas nuevas que devuelve la probabilidad de cancelación, lista para integrarse en un flujo operativo.

## Cómo ejecutar

1. Instalar dependencias:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
2. Abrir `cancelacion_tiquetes.ipynb` y ejecutar las celdas en orden (el archivo `train.csv` debe estar en la misma carpeta).

## Estructura

```
ModeloCV/
├── cancelacion_tiquetes.ipynb   # Notebook con todo el análisis y el modelo
├── train.csv                    # Dataset de reservas
└── README.md
```
