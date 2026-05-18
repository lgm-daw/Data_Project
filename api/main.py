from fastapi import FastAPI
import joblib
import pandas as pd

from database.mongo import predictions_collection

from datetime import datetime

app = FastAPI()

# cargar modelo
model = joblib.load("models/modelo_final.pkl")


@app.get("/")
def home():
    return {"message": "API funcionando correctamente"}


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    result = {
        "input": data,
        "prediction": int(prediction[0]),
        "model": "RandomForest",
        "timestamp": datetime.utcnow()
    }

    # guardar en Mongo
    inserted = predictions_collection.insert_one(result)

    return {
        "id": str(inserted.inserted_id),
        "prediction": int(prediction[0])
    }