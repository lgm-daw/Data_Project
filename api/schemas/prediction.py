from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class PredictionInput(BaseModel):
    """Schema para los datos de entrada de la predicción.

    Campos del Titanic:
    - pclass: Clase del pasajero (1, 2, 3)
    - sex: Sexo (male, female)
    - age: Edad
    - sibsp: Número de hermanos/esposos a bordo
    - parch: Número de padres/hijos a bordo
    - fare: Tarifa pagada
    - embarked: Puerto de embarque (C, Q, S)
    """
    pclass: int
    sex: str
    age: float
    sibsp: int
    parch: int
    fare: float
    embarked: str


class PredictionResponse(BaseModel):
    """Schema para la respuesta de predicción."""
    id: str
    prediction: int
    probability: Optional[float] = None
    timestamp: datetime
    model_used: str


class PredictionHistory(BaseModel):
    """Schema para el historial de predicciones."""
    id: str
    input_data: dict
    prediction: int
    timestamp: datetime
    model_used: str