import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline

from src.features.build_features import build_features, build_preprocessor, load_data, split_data
from src.utils.config import MLFLOW_TRACKING_URI, RANDOM_STATE


def save_evaluation_plot(pipeline, X_test, y_test, predictions, r2, output_path):
    """Write predicted-vs-actual and feature-importance figure for the README."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]
    names = [name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()]
    importances = regressor.feature_importances_
    top = np.argsort(importances)[-10:]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))
    fig.suptitle("Biogas production forecasting (synthetic digester data)", fontsize=13)

    ax = axes[0]
    ax.scatter(y_test, predictions, s=14, alpha=0.55, c="#2e8b57", edgecolors="none")
    lo = float(min(np.min(y_test), np.min(predictions)))
    hi = float(max(np.max(y_test), np.max(predictions)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Actual biogas production (m³/day)")
    ax.set_ylabel("Predicted (m³/day)")
    ax.set_title(f"Gradient Boosting, test set (R² = {r2:.3f})")

    ax = axes[1]
    ax.barh(np.array(names)[top], importances[top], color="#2f6db3")
    ax.set_title("Top 10 feature importances")

    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Evaluation plot saved to {output_path}")


def train_model():
    """Train a biogas production forecasting model and log to MLflow."""
    mlflow_available = False
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("biogas-production-forecasting")
        mlflow_available = True
    except Exception as e:
        print(f"Warning: MLflow server not available ({e}). Training without remote tracking.")
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("biogas-production-forecasting")

    df = load_data()
    df = build_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor()
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", model)]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    rmse = root_mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Save model locally
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "biogas_model.pkl"
    joblib.dump(pipeline, model_path)

    # Save metrics
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }
    with open(models_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    try:
        save_evaluation_plot(
            pipeline,
            X_test,
            y_test,
            predictions,
            r2,
            Path("assets/biogas_pred_vs_actual.png"),
        )
    except Exception as exc:
        print(f"Warning: could not save evaluation plot ({exc}).")

    if mlflow_available:
        try:
            with mlflow.start_run(run_name="gradient-boosting-v1") as run:
                mlflow.log_param("model_type", "GradientBoostingRegressor")
                mlflow.log_param("n_estimators", model.n_estimators)
                mlflow.log_param("learning_rate", model.learning_rate)
                mlflow.log_param("max_depth", model.max_depth)
                mlflow.log_param("random_state", RANDOM_STATE)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)
                mlflow.sklearn.log_model(pipeline, "model")
                mlflow.log_artifact(str(model_path), artifact_path="artifacts")
                print(f"MLflow Run ID: {run.info.run_id}")
        except Exception as e:
            print(f"Warning: Could not log to MLflow ({e}). Model and metrics saved locally.")

    print(f"Model trained successfully")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_model()
