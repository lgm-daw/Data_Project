import os
from pymongo import MongoClient
from datetime import datetime


class MongoDB:
    """Gestión de conexión y operaciones con MongoDB."""

    def __init__(self):
        # URI desde variable de entorno (configurada en docker-compose)
        self.mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.db_name = "predictions_db"
        self.collection_name = "predictions"

        # Conectar
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]

    def save_prediction(
        self,
        input_data: dict,
        prediction: int,
        model_used: str,
        timestamp: datetime
    ) -> str:
        """Guarda una predicción y devuelve su ID."""
        document = {
            "input": input_data,
            "prediction": prediction,
            "model": model_used,
            "timestamp": timestamp
        }
        result = self.collection.insert_one(document)
        return str(result.inserted_id)

    def get_all_predictions(self, limit: int = 50) -> list:
        """Recupera todas las predicciones, las más recientes primero."""
        predictions = list(
            self.collection
            .find()
            .sort("timestamp", -1)
            .limit(limit)
        )
        # Convertir ObjectId a string
        for p in predictions:
            p["_id"] = str(p["_id"])
        return predictions

    def get_prediction_by_id(self, prediction_id: str) -> dict:
        """Recupera una predicción por su ID."""
        from bson import ObjectId
        prediction = self.collection.find_one({"_id": ObjectId(prediction_id)})
        if prediction:
            prediction["_id"] = str(prediction["_id"])
        return prediction