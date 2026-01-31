Despliegue de Modelo ML: Detección y predicción de Diabetes

Este proyecto consiste en el despliegue de un modelo de Machine Learning para la detección de diabetes, desarrollado para el curso de Cloud Computing del Master en Data Science.

Profesor:

- Ahmad Armoush

Alumnos:

- Eyber Cárdenas
- Karen Ramírez
- Valentina Ariztía
- Joe Armijo

1. Descripción del Problema

El objetivo es proporcionar una herramienta de predicción basada en el dataset "Diabetes". El modelo utiliza variables clínicas (glucosa, presión arterial, IMC, etc.) para clasificar si un paciente tiene tendencia a desarrollar diabetes.

- Modelo: Clasificación Random Forest + Standard Scaler.
- Métrica de Evaluación: Recall, Score F1, Accuracy, Curva de ROC.
- Dataset: Diabetes Health Indicators
- Fuente: https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset
- Librerías: SkLearn, FastAPI, Mathplotlib, Seaborn, Pickle, Pandas.

2. Estructura del Proyecto

El repositorio sigue la estructura obligatoria para un despliegue exitoso sin Docker:

- app.py: Código fuente de la API desarrollado con FastAPI.
- entrenamiento.py: Script utilizado para el entrenamiento y exportación del modelo.
- model.pkl: Serialización del modelo entrenado listo para producción.
- requirements.txt: Lista de dependencias del proyecto.
- runtime.txt: Especificación de la versión de Python (3.9.17).
- Procfile: Instrucciones de ejecución para el servidor Uvicorn.
- python.version: Especificación de la versión de Python (3.9.17).
- scaler.pkl: serialización (persistencia) de un objeto StandardScaler de la librería Scikit-Learn, el cual ha sido previamente ajustado (fitted) con la distribución estadística del dataset de entrenamiento.
- metrics.json: resultado de las salidas del modelo.
- gitignore: para no considerar algunos archivos por GitHub al momento de cargar (ejemplo: .txt)

3. Instrucciones para Ejecución Local

La API está desarrollada utilizando el framework FastAPI y es ejecutada mediante
el servidor ASGI Uvicorn.

Para correr la API en tu entorno local, sigue estos pasos:

    a. Clonar el repositorio.

    b. Instalar dependencias y librerías necesarias:

        Bash
        pip install -r requirements.txt


    c. Ejecutar el servidor

        Bash
      uvicorn app:app --reload

    d. Acceder a http://127.0.0.1:8000/docs para ver la documentación interactiva.

4. Ejemplo de request al endpoint /predict

   POST /predict

- JSON (por default)

  {
  "hba1c": 5.5,
  "glucose_postprandial": 140,
  "glucose_fasting": 90,
  "age": 35,
  "bmi": 24.5,
  "systolic_bp": 120,
  "cholesterol_total": 180,
  "physical_activity_minutes_per_week": 150
  }

5. Plataforma cloud usada para el deploy

- GCP Cloud Run (modo no Docker)

6. Instrucciones para Ejecución en la nube

La aplicación se encuentra desplegada en la nube GCP y accesible mediante una URL pública, cumpliendo con el flujo de despliegue exigido en la pauta del curso.

El despliegue se realizó utilizando:

- Runtime de Python especificado en el archivo runtime.txt.
- Archivo Procfile para la ejecución del servidor ASGI Uvicorn.
- Carga del modelo entrenado desde archivos .pkl al iniciar la aplicación.

Flujo de despliegue:

1. El repositorio GitHub fue conectado a la plataforma cloud utilizada.
2. La plataforma utiliza el runtime de Python definido en runtime.txt.
3. El servidor se levanta mediante Uvicorn según lo especificado en el Procfile.
4. La API queda expuesta a través de una URL pública.

Una vez desplegada, el endpoint POST /predict se encuentra operativo y accesible
desde la nube, permitiendo realizar predicciones en tiempo real.
La URL de la API LIVE es entregada junto al repositorio GitHub vía el LMS.

Ejemplos de respuesta de la API (/predict) para el modelo desplegado que debería ver.

Ejemplo 1: Paciente clasificado como SANO:

Request (JSON):

{
"hba1c": 5.8,
"glucose_postprandial": 145,
"glucose_fasting": 110,
"age": 40,
"bmi": 27.5,
"systolic_bp": 130,
"cholesterol_total": 200,
"physical_activity_minutes_per_week": 30
}

Response (JSON):

{
"resultado_diagnostico": "Negativo (Sano)",
"probabilidad_calculada": 0.1101,
"nivel_alerta": "BAJA",
"mensaje_clinico": "No se detectan patrones de riesgo significativos.",
"factores_de_riesgo": [
"HbA1c Elevada (5.8%) - Rango de Prediabetes.",
"Glucosa Ayunas Alterada (110 mg/dL).",
"Sobrepeso (IMC: 27.5)."
],
"meta_info": {
"rango_prediabetes": "0.3 - 0.4798",
"umbral_diabetes": 0.4798
}
}

Ejemplo 2: Paciente clasificado con Diabetes / Riesgo Alto:

Request (JSON):
{
"hba1c": 6.8,
"glucose_postprandial": 175,
"glucose_fasting": 110,
"age": 40,
"bmi": 40.5,
"systolic_bp": 130,
"cholesterol_total": 220,
"physical_activity_minutes_per_week": 0
}

Response (JSON):
{
"resultado_diagnostico": "Positivo (Diabetes Tipo 2)",
"probabilidad_calculada": 0.9605,
"nivel_alerta": "ALTA",
"mensaje_clinico": "El modelo detecta patrones clínicos consistentes con diabetes.",
"factores_de_riesgo": [
"HbA1c Crítica (6.8%) - Rango diabético.",
"Glucosa Ayunas Alterada (110 mg/dL).",
"Obesidad (IMC: 40.5)."
],
"meta_info": {
"rango_prediabetes": "0.3 - 0.4798",
"umbral_diabetes": 0.4798
}
}

7. URL de repositorio GITHUB

https://github.com/ejcc1991/TareaGrupal_CC_Diabetes_ML/

8. URL para despliegue en Google Cloud Run (GCP)

https://tareagrupal-cc-diabetes-ml-197521535572.us-central1.run.app/docs#/
