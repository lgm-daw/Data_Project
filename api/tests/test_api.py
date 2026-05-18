from fastapi.testclient import TestClient
import sys
import os

# Añadir directorio api al path para poder importar main
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import app

client = TestClient(app)


def test_root():
    """Test: Verificar que la raíz responde correctamente."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "docs" in response.json()


def test_health():
    """Test: Verificar endpoint de salud."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_valid_data():
    """Test: Predicción con datos válidos."""
    valid_input = {
        "pclass": 1,
        "sex": "female",
        "age": 29,
        "sibsp": 0,
        "parch": 0,
        "fare": 100,
        "embarked": "C"
    }
    response = client.post("/predict", json=valid_input)
    assert response.status_code == 200
    json_resp = response.json()
    assert "prediction" in json_resp
    assert "id" in json_resp
    assert json_resp["prediction"] in [0, 1]


def test_predict_invalid_data():
    """Test: Predicción con datos incompletos debe devolver error 422."""
    invalid_input = {
        "pclass": 1,
        "sex": "female"
        # Faltan campos obligatorios
    }
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422


def test_get_predictions():
    """Test: Verificar que el endpoint de historial funciona."""
    response = client.get("/predictions")
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert isinstance(response.json()["predictions"], list)


def test_get_prediction_not_found():
    """Test: Buscar una predicción con ID inexistente debe devolver 404."""
    fake_id = "507f1f77bcf86cd799439011"  # ObjectId válido pero inexistente
    response = client.get(f"/predictions/{fake_id}")
    assert response.status_code == 404
