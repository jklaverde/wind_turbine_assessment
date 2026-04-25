import os
import tempfile

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.training.train import FEATURES, TARGET, train_and_save


def _make_cleaned_data(tmp_dir: str) -> None:
    """Write minimal train.csv and test.csv under tmp_dir/data/cleaned/."""
    out = os.path.join(tmp_dir, "data", "cleaned")
    os.makedirs(out, exist_ok=True)

    rows = []
    for t, h, n, f in [
        (95, 88, 78, 1),
        (40, 50, 50, 0),
        (92, 87, 76, 1),
        (35, 30, 40, 0),
        (98, 90, 80, 1),
        (45, 60, 55, 0),
    ]:
        rows.append({"temperature": t, "humidity": h, "noise_level": n, "failure": f})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "train.csv"), index=False)
    df.to_csv(os.path.join(out, "test.csv"),  index=False)


class TestTrainAndSave:
    def test_model_file_created(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _make_cleaned_data(str(tmp_path))
        train_and_save()
        assert os.path.exists("model/wind_turbine_model.joblib")

    def test_saved_object_is_random_forest(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _make_cleaned_data(str(tmp_path))
        train_and_save()
        model = joblib.load("model/wind_turbine_model.joblib")
        assert isinstance(model, RandomForestClassifier)

    def test_model_can_predict(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _make_cleaned_data(str(tmp_path))
        train_and_save()
        model = joblib.load("model/wind_turbine_model.joblib")
        pred = model.predict([[95, 88, 78]])
        assert pred[0] in (0, 1)

    def test_model_returns_probabilities(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _make_cleaned_data(str(tmp_path))
        train_and_save()
        model = joblib.load("model/wind_turbine_model.joblib")
        proba = model.predict_proba([[95, 88, 78]])
        assert proba.shape[1] == 2
        assert abs(proba[0].sum() - 1.0) < 1e-6
