from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

app = FastAPI(
    title="Sistema de Detección de Diabetes",
    description="API de predicción con soporte para escalado de datos (StandardScaler)",
    version="1.1.0"
)

# --- CARGA DE ACTIVOS (Modelo y Escalador) ---
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

model = None
scaler = None

try:
    # Cargamos el Modelo
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("✅ Modelo cargado exitosamente.")
    
    # Cargamos el Escalador (Pieza clave para evitar el 'Siempre Riesgo Alto')
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print("✅ Escalador cargado exitosamente.")
    else:
        print("⚠️ Advertencia: No se encontró 'scaler.pkl'. Las predicciones podrían ser incorrectas.")

except Exception as e:
    print(f"❌ Error crítico al cargar activos: {str(e)}")

# --- DEFINICIÓN DEL ESQUEMA DE DATOS ---
class InputData(BaseModel):
    hba1c: float = Field(..., ge=0, le=20, example=5.5)
    glucose_postprandial: int = Field(..., ge=0, le=500, example=140)
    glucose_fasting: int = Field(..., ge=0, le=500, example=90)
    age: int = Field(..., ge=0, le=120, example=35)
    bmi: float = Field(..., ge=10, le=60, example=24.5)
    systolic_bp: int = Field(..., ge=50, le=250, example=120)
    cholesterol_total: int = Field(..., ge=50, le=500, example=180)
    physical_activity_minutes_per_week: int = Field(..., ge=0, le=10080, example=150)

# --- RUTAS ---
@app.get("/")
def home():
    return {
        "status": "online",
        "model_ready": model is not None,
        "scaler_ready": scaler is not None,
        "message": "API lista para recibir datos en /predict"
    }

@app.post("/predict")
def predict(data: InputData):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="El modelo o el escalador no están disponibles.")
    
    try:
        # 1. Convertir entrada a DataFrame (Nombres de columnas deben coincidir con el entrenamiento)
        df = pd.DataFrame([data.dict()])
        
        # 2. ESCALAR los datos (Esto es lo que corrige el error de 'Riesgo Alto' constante)
        # El scaler convierte tus números a la escala que el modelo entiende (ej: de 0 a 1)
        df_scaled = scaler.transform(df)
        
        # 3. Realizar la predicción
        prediction = model.predict(df_scaled)
        probability = model.predict_proba(df_scaled)[0][1] if hasattr(model, "predict_proba") else None
        
        # 4. Resultado amigable
        result = "Positivo (Riesgo Alto)" if prediction[0] == 1 else "Negativo (Riesgo Bajo)"
        
        return {
            "prediccion": result,
            "probabilidad": round(float(probability), 4) if probability is not None else "N/A",
            "metodo": "Random Forest + StandardScaler"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el proceso de predicción: {str(e)}")