from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()


# Cargar modelo al arrancar
model = joblib.load("models/modelo_final.pkl")


@app.get("/")
def root():
    return {"message": "API funcionando correctamente"}


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    return {
        "prediction": int(prediction[0])
    }