import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


def load_data(path: Path = Path("data/biogas_production.csv")) -> pd.DataFrame:
    """Load the biogas dataset."""
    return pd.read_csv(path)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based and interaction features."""
    df = df.copy()
    if "date" not in df.columns:
        df["date"] = pd.Timestamp.now().normalize()
    else:
        df["date"] = pd.to_datetime(df["date"])

    # Time-based features
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek

    # Cyclical encoding for day_of_year
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # Interaction features
    df["load_x_temp"] = df["organic_load"] * df["temperature"]
    df["ph_deviation"] = np.abs(df["ph"] - 7.25)

    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split data into train and test sets (time-based split is preferred, using random for simplicity)."""
    features = [
        "temperature",
        "ph",
        "organic_load",
        "hydraulic_retention_time",
        "substrate_type",
        "humidity",
        "ambient_temperature",
        "previous_day_production",
        "day_of_year_sin",
        "day_of_year_cos",
        "month",
        "load_x_temp",
        "ph_deviation",
    ]
    target = "biogas_production"

    X = df[features]
    y = df[target]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_preprocessor() -> ColumnTransformer:
    """Build sklearn preprocessor for numeric and categorical features."""
    numeric_features = [
        "temperature",
        "ph",
        "organic_load",
        "hydraulic_retention_time",
        "humidity",
        "ambient_temperature",
        "previous_day_production",
        "day_of_year_sin",
        "day_of_year_cos",
        "month",
        "load_x_temp",
        "ph_deviation",
    ]
    categorical_features = ["substrate_type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    return preprocessor


if __name__ == "__main__":
    import numpy as np

    df = load_data()
    df = build_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(X_train.head())
