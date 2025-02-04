# Tarea 2: Estimación de Precios de Bienes Raíces

Este proyecto es una Prueba de Concepto (PoC) para estimar el precio de una propiedad basado en características como el tamaño, número de habitaciones, baños y ubicación. El objetivo es proporcionar una herramienta que permita a los usuarios obtener una estimación rápida del valor de una casa.

## Descripción del Proyecto

El proyecto utiliza un conjunto de datos de precios de venta de casas en Ames, Iowa, para entrenar un modelo de machine learning que predice el precio de una propiedad. El flujo de trabajo incluye:

1. **Exploración de Datos (EDA):** Análisis de las características del dataset, identificación de valores faltantes y visualización de relaciones entre variables.
2. **Preprocesamiento de Datos:** Limpieza, codificación de variables categóricas y escalado de características.
3. **Selección de Variables:** Uso de técnicas para seleccionar las características más relevantes.
4. **Entrenamiento del Modelo:** Entrenamiento de un modelo de regresión (Random Forest) para predecir el precio.
5. **Evaluación del Modelo:** Medición del rendimiento del modelo usando métricas como RMSE y R².
6. **Inferencia:** Implementación de una función para realizar predicciones basadas en entradas del usuario.

## Requisitos

Para ejecutar este proyecto, necesitas:

- Python 3.8 o superior.
- Bibliotecas de Python listadas en `requirements.txt`.

## Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu-usuario/estimacion-precios-bienes-raices.git
   cd estimacion-precios-bienes-raices
   ```

2. Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

1. Abre en el Jupyter Notebook `modelo.py`:

2. Sigue las celdas del notebook para:

    * Explorar el dataset.
    * Preprocesar los datos.
    * Entrenar y evaluar el modelo.
    * Realizar inferencias con nuevas entradas.

3. Para realizar una predicción, usa la función `predict_price`:
    ```python
    predicted_price = predict_price(sqft_living=2000, bedrooms=3, bathrooms=2, zipcode=98001)
    print(f"Precio estimado: ${predicted_price:.2f}")
    ```
## Estructura del Proyecto

estimacion-precios-bienes-raices/
├── notebooks/               # Carpeta para los notebooks
│   └── modelo.py            # Notebook principal
├── requirements.txt         # Dependencias del proyecto
├── README.md                # Este archivo
└── diagrams/                # Diagramas del flujo de trabajo
    └── workflow.png         # Diagrama del flujo de trabajo