from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas.prediction import PredictionInput, PredictionResponse
from services.predictor import TitanicPredictor
from database.mongo import MongoDB

# Inicializar FastAPI
app = FastAPI(
    title="Titanic Survival Predictor",
    description="API para predecir la supervivencia de pasajeros del Titanic usando ML",
    version="1.0.0"
)

# CORS (permite peticiones desde cualquier origen)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo UNA sola vez al arrancar
predictor = TitanicPredictor()

# Conectar a MongoDB UNA sola vez al arrancar
db = MongoDB()


@app.get("/")
async def root():
    """Redirige a la documentación Swagger."""
    return {
        "message": "Titanic Survival Prediction API",
        "docs": "/docs",
        "endpoints": {
            "POST /predict": "Realizar una predicción",
            "GET /predictions": "Ver historial de predicciones",
            "GET /predictions/{id}": "Ver una predicción específica"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput):
    """Recibe datos de un pasajero y devuelve predicción de supervivencia."""
    try:
        # Convertir a diccionario
        data = input_data.model_dump()

        # Realizar predicción
        result = predictor.predict(data)

        # Guardar en MongoDB
        prediction_id = db.save_prediction(
            input_data=data,
            prediction=result["prediction"],
            model_used=result["model_used"],
            timestamp=result["timestamp"]
        )

        # Devolver respuesta
        return PredictionResponse(
            id=prediction_id,
            prediction=result["prediction"],
            probability=result["probability"],
            timestamp=result["timestamp"],
            model_used=result["model_used"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")


@app.get("/predictions")
async def get_predictions(limit: int = 50):
    """Devuelve el historial de predicciones almacenadas en MongoDB."""
    try:
        predictions = db.get_all_predictions(limit=limit)
        return {"count": len(predictions), "predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener predicciones: {str(e)}")


@app.get("/predictions/{prediction_id}")
async def get_prediction(prediction_id: str):
    """Devuelve una predicción específica por su ID."""
    try:
        prediction = db.get_prediction_by_id(prediction_id)
        if prediction is None:
            raise HTTPException(status_code=404, detail="Predicción no encontrada")
        return prediction
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/health")
async def health_check():
    """Endpoint de salud para verificar que la API está funcionando."""
    return {"status": "healthy", "model": predictor.model_name}