from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

app = FastAPI(
    title="API de Predicción de Diabetes",
    description="Sistema de detección de riesgo basado en Machine Learning"
)

# --- CARGA DEL MODELO ---
# Usamos una ruta absoluta para evitar confusiones en el servidor
MODEL_PATH = os.path.join(os.getcwd(), "model.pkl")
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("✅ Modelo cargado exitosamente desde:", MODEL_PATH)
    else:
        print(f"❌ ERROR: El archivo {MODEL_PATH} no existe en el directorio.")
except Exception as e:
    print(f"❌ ERROR CRÍTICO al cargar el modelo: {str(e)}")

# --- DEFINICIÓN DE DATOS ---
class InputData(BaseModel):
    hba1c: float = Field(..., description="Hemoglobina Glicosilada (4-20%)", ge=0, le=20, example=5.5)
    glucose_postprandial: int = Field(..., description="Glucosa 2h post comida", ge=0, le=500, example=140)
    glucose_fasting: int = Field(..., description="Glucosa en ayunas", ge=0, le=500, example=90)
    age: int = Field(..., description="Edad", ge=0, le=120, example=35)
    bmi: float = Field(..., description="IMC (Peso/Altura²)", ge=10, le=60, example=24.5)
    systolic_bp: int = Field(..., description="Presión arterial sistólica", ge=50, le=250, example=120)
    cholesterol_total: int = Field(..., description="Colesterol total", ge=50, le=500, example=180)
    physical_activity_minutes_per_week: int = Field(..., description="Ejercicio semanal (min)", ge=0, le=10080, example=150)

# --- RUTAS ---
@app.get("/")
def home():
    """Ruta de verificación de salud para Google Cloud Run"""
    return {
        "status": "online",
        "model_loaded": model is not None,
        "message": "API de Predicción de Diabetes funcionando correctamente"
    }

@app.post("/predict")
def predict(data: InputData):
    """Realiza la predicción basada en los datos del paciente"""
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible en el servidor.")
    
    try:
        # Convertir datos a DataFrame con los nombres de columnas correctos
        df = pd.DataFrame([data.dict()])
        
        # Realizar la predicción
        prediction = model.predict(df)
        probability = model.predict_proba(df)[0][1] if hasattr(model, "predict_proba") else None
        
        # Formatear respuesta
        result = "Positivo (Riesgo Alto)" if prediction[0] == 1 else "Negativo (Riesgo Bajo)"
        
        return {
            "prediccion": result,
            "probabilidad_diabetes": round(float(probability), 4) if probability is not None else "N/A",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")