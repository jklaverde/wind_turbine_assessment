from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_REQUEST = {
    "turbine_id": "TRB-001",
    "measurement_timestamp": "20240115_103000",
    "temperature": 95.0,
    "humidity": 88.0,
    "noise_level": 78.0,
}


def _mock_model(probability: float = 0.87):
    model = MagicMock()
    model.predict_proba.return_value = np.array([[1 - probability, probability]])
    return model


@pytest.fixture()
def client_with_model():
    with patch("src.api.main._model", _mock_model()):
        yield TestClient(app)


@pytest.fixture()
def client_no_model():
    with patch("src.api.main._model", None):
        yield TestClient(app)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_status_ok(self, client_with_model):
        r = client_with_model.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_model_loaded_true(self, client_with_model):
        r = client_with_model.get("/api/v1/health")
        assert r.json()["model_loaded"] is True

    def test_model_loaded_false(self, client_no_model):
        r = client_no_model.get("/api/v1/health")
        assert r.json()["model_loaded"] is False

    def test_timestamp_present(self, client_with_model):
        r = client_with_model.get("/api/v1/health")
        assert "timestamp" in r.json()


# ---------------------------------------------------------------------------
# Single predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_success_200(self, client_with_model):
        r = client_with_model.post("/api/v1/predict", json=VALID_REQUEST)
        assert r.status_code == 200

    def test_response_fields(self, client_with_model):
        r = client_with_model.post("/api/v1/predict", json=VALID_REQUEST)
        body = r.json()
        assert body["turbine_id"] == "TRB-001"
        assert body["measurement_timestamp"] == "20240115_103000"
        assert "response_timestamp" in body
        assert isinstance(body["potential_anomaly"], bool)
        assert 0.0 <= body["probability"] <= 1.0

    def test_high_probability_is_anomaly(self, client_with_model):
        with patch("src.api.main._model", _mock_model(0.87)):
            r = TestClient(app).post("/api/v1/predict", json=VALID_REQUEST)
        assert r.json()["potential_anomaly"] is True

    def test_low_probability_is_not_anomaly(self):
        with patch("src.api.main._model", _mock_model(0.20)):
            r = TestClient(app).post("/api/v1/predict", json=VALID_REQUEST)
        assert r.json()["potential_anomaly"] is False

    def test_model_not_ready_503(self, client_no_model):
        r = client_no_model.post("/api/v1/predict", json=VALID_REQUEST)
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "MODEL_NOT_READY"

    def test_missing_field_422(self, client_with_model):
        bad = {k: v for k, v in VALID_REQUEST.items() if k != "temperature"}
        r = client_with_model.post("/api/v1/predict", json=bad)
        assert r.status_code == 422

    def test_humidity_out_of_range_422(self, client_with_model):
        bad = {**VALID_REQUEST, "humidity": 150.0}
        r = client_with_model.post("/api/v1/predict", json=bad)
        assert r.status_code == 422

    def test_negative_temperature_422(self, client_with_model):
        bad = {**VALID_REQUEST, "temperature": -1.0}
        r = client_with_model.post("/api/v1/predict", json=bad)
        assert r.status_code == 422

    def test_invalid_timestamp_format_422(self, client_with_model):
        bad = {**VALID_REQUEST, "measurement_timestamp": "2024-01-15 10:30:00"}
        r = client_with_model.post("/api/v1/predict", json=bad)
        assert r.status_code == 422

    def test_empty_turbine_id_422(self, client_with_model):
        bad = {**VALID_REQUEST, "turbine_id": ""}
        r = client_with_model.post("/api/v1/predict", json=bad)
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Batch predict
# ---------------------------------------------------------------------------

class TestPredictBatch:
    def test_all_valid_returns_200(self, client_with_model):
        r = client_with_model.post("/api/v1/predict/batch", json=[VALID_REQUEST, VALID_REQUEST])
        assert r.status_code == 200

    def test_summary_counts(self, client_with_model):
        bad_item = {**VALID_REQUEST, "humidity": 999.0}
        r = client_with_model.post("/api/v1/predict/batch", json=[VALID_REQUEST, bad_item])
        body = r.json()
        assert body["total"] == 2
        assert body["succeeded"] == 1
        assert body["failed"] == 1

    def test_result_indices(self, client_with_model):
        r = client_with_model.post("/api/v1/predict/batch", json=[VALID_REQUEST, VALID_REQUEST])
        indices = [item["index"] for item in r.json()["results"]]
        assert indices == [0, 1]

    def test_failed_item_has_error_key(self, client_with_model):
        bad_item = {**VALID_REQUEST, "humidity": 999.0}
        r = client_with_model.post("/api/v1/predict/batch", json=[bad_item])
        result = r.json()["results"][0]
        assert "error" in result
        assert result["error"]["code"] == "VALIDATION_ERROR"

    def test_batch_size_exceeded_422(self, client_with_model):
        with patch("src.api.main.MAX_BATCH_SIZE", 2):
            r = client_with_model.post(
                "/api/v1/predict/batch",
                json=[VALID_REQUEST, VALID_REQUEST, VALID_REQUEST],
            )
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "BATCH_SIZE_EXCEEDED"

    def test_model_not_ready_503(self, client_no_model):
        r = client_no_model.post("/api/v1/predict/batch", json=[VALID_REQUEST])
        assert r.status_code == 503

    def test_empty_batch_returns_zero_counts(self, client_with_model):
        r = client_with_model.post("/api/v1/predict/batch", json=[])
        body = r.json()
        assert body["total"] == 0
        assert body["succeeded"] == 0
        assert body["failed"] == 0
