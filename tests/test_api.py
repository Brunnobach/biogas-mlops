from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

PAYLOAD = {
    "temperature": 37.0,
    "ph": 7.2,
    "organic_load": 3.5,
    "hydraulic_retention_time": 25,
    "substrate_type": "cattle_swine_poultry",
    "humidity": 85,
    "ambient_temperature": 22,
    "previous_day_production": 120,
    "month": 1,
}


@pytest.fixture
def client():
    import src.api.app as app_module

    model_path = Path("models/biogas_model.pkl")
    if not model_path.exists():
        pytest.skip("trained model is required for API tests")
    if app_module.model is None:
        app_module.model = joblib.load(model_path)
    return TestClient(app_module.app)


def test_predict_uses_request_month(client):
    january = client.post("/predict", json={**PAYLOAD, "month": 1})
    july = client.post("/predict", json={**PAYLOAD, "month": 7})

    assert january.status_code == 200, january.text
    assert july.status_code == 200, july.text
    assert "predicted_biogas_production_m3" in january.json()
    assert "predicted_biogas_production_m3" in july.json()
