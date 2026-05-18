import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime


class TitanicPredictor:
    """Carga el modelo y scaler, y realiza predicciones sobre datos del Titanic."""

    def __init__(self):
        # Rutas a los archivos del modelo
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_path = os.path.join(base_dir, "notebook", "models", "modelo_final.pkl")
        scaler_path = os.path.join(base_dir, "notebook", "models", "scaler.pkl")

        # Cargar modelo y scaler
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.model_name = "Random Forest"

        # Lista de columnas que espera el modelo
        self.feature_columns = [
            'pclass', 'age', 'sibsp', 'parch', 'fare',
            'sex_male', 'embarked_Q', 'embarked_S'
        ]



    def preprocess(self, input_data: dict) -> np.ndarray:
        """Transforma los datos de entrada al formato que espera el modelo."""
        # Convertir a DataFrame
        df = pd.DataFrame([input_data])

        # Crear dummies para sex (male=1, female=0)
        df['sex_male'] = (df['sex'] == 'male').astype(int)

        # Crear dummies para embarked
        df['embarked_Q'] = (df['embarked'] == 'Q').astype(int)
        df['embarked_S'] = (df['embarked'] == 'S').astype(int)

        # Eliminar columnas originales que no usa el modelo
        df = df.drop(['sex', 'embarked'], axis=1)

        # Asegurar que tiene todas las columnas en el orden correcto
        df_processed = df[self.feature_columns]

        # Escalar
        df_scaled = self.scaler.transform(df_processed)

        return df_scaled

    def predict(self, input_data: dict) -> dict:
        """Realiza predicción y devuelve resultado con metadatos."""
        processed_data = self.preprocess(input_data)

        prediction = int(self.model.predict(processed_data)[0])

        # Probabilidad de supervivencia (clase 1)
        probability = self.model.predict_proba(processed_data)[0][1]

        return {
            "prediction": prediction,
            "probability": round(float(probability), 4),
            "timestamp": datetime.utcnow(),
            "model_used": self.model_name
        }