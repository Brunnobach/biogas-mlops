import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.generate_dataset import generate_biogas_data
from src.features.build_features import build_features, load_data, split_data, build_preprocessor


def test_dataset_generation():
    df = generate_biogas_data(n_samples=100)
    assert len(df) == 100
    assert "biogas_production" in df.columns
    assert "previous_day_production" in df.columns
    assert df["biogas_production"].min() >= 20
    assert df["biogas_production"].max() <= 500


def test_feature_building():
    df = generate_biogas_data(n_samples=100)
    df = build_features(df)
    assert "day_of_year_sin" in df.columns
    assert "day_of_year_cos" in df.columns
    assert "load_x_temp" in df.columns
    assert "ph_deviation" in df.columns


def test_preprocessor():
    df = generate_biogas_data(n_samples=100)
    df = build_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    assert X_train_processed.shape[0] == len(X_train)
    assert X_train_processed.shape[1] > 0


def test_data_split():
    df = generate_biogas_data(n_samples=100)
    df = build_features(df)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
    assert len(X_test) == 20
    assert len(X_train) == 80


def test_build_features_uses_provided_month_when_date_is_missing():
    """API requests send month but not date; month must not be replaced by today."""
    requested_month = 1
    df = pd.DataFrame(
        [
            {
                "temperature": 37.0,
                "ph": 7.2,
                "organic_load": 3.5,
                "hydraulic_retention_time": 25,
                "substrate_type": "cattle_swine_poultry",
                "humidity": 85,
                "ambient_temperature": 22,
                "previous_day_production": 120,
                "month": requested_month,
            }
        ]
    )

    out = build_features(df)

    assert int(out["month"].iloc[0]) == requested_month
    # Mid-January day-of-year, not "today"
    assert 1 <= int(out["day_of_year"].iloc[0]) <= 31


def test_build_features_keeps_month_from_date_during_training():
    df = generate_biogas_data(n_samples=30, start_date="2022-03-01")
    out = build_features(df)
    assert int(out["month"].iloc[0]) == 3
    assert int(out.loc[out.index[-1], "month"]) == 3


def test_build_features_month_changes_cyclical_encoding():
    def payload(month: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "temperature": 37.0,
                    "ph": 7.2,
                    "organic_load": 3.5,
                    "hydraulic_retention_time": 25,
                    "substrate_type": "cattle_swine_poultry",
                    "humidity": 85,
                    "ambient_temperature": 22,
                    "previous_day_production": 120,
                    "month": month,
                }
            ]
        )

    january = build_features(payload(1))
    july = build_features(payload(7))
    assert january["day_of_year_sin"].iloc[0] != july["day_of_year_sin"].iloc[0]
    assert int(january["month"].iloc[0]) == 1
    assert int(july["month"].iloc[0]) == 7
