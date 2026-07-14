# Biogas MLOps

**End-to-end MLOps pipeline for biogas production forecasting**

A production-ready machine learning pipeline that predicts biogas production from operational data, with experiment tracking, model registry, containerized API and CI/CD.

---

## 🌐 Live Portfolio

📊 **Project documentation and live dashboard:** https://brunnobach.github.io/biogas-mlops/

---

## 🎯 What this project demonstrates

| Skill | How it is applied here |
|-------|------------------------|
| **MLOps** | MLflow tracking, model registry, reproducible pipelines |
| **Cloud-ready ML** | Dockerized FastAPI inference API |
| **CI/CD** | GitHub Actions for testing and building |
| **Domain expertise** | Biogas production forecasting |
| **Software engineering** | Modular code, tests, logging |

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Raw Data      │────▶│  Feature Eng.   │────▶│  Model Training │
│  (CSV / API)    │     │   (Pandas)      │     │  (scikit-learn) │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌────────────────────────────────────────────────────────────────┐
│                     MLflow Tracking + Registry                    │
└────────────────────────────────────────────────────────────────┘
                                                         │
                                                         ▼
                                              ┌─────────────────┐
                                              │  FastAPI API     │
                                              │  /predict        │
                                              │  (Docker)        │
                                              └─────────────────┘
```

---

## 🛠️ Tech Stack

- Python 3.10+
- scikit-learn
- MLflow
- FastAPI
- Docker
- GitHub Actions
- pytest

---

## 📁 Project Structure

```
biogas-mlops/
├── .github/workflows/    # CI/CD pipelines
├── data/                 # Raw and processed data
├── notebooks/            # EDA and experiments
├── src/
│   ├── features/         # Feature engineering
│   ├── models/           # Training and inference
│   └── api/              # FastAPI app
├── tests/                # Unit tests
├── mlruns/               # MLflow artifacts (local)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/Brunnobach/biogas-mlops.git
cd biogas-mlops

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train model
python src/models/train.py

# Start MLflow UI
mlflow ui

# Run API locally
python src/api/app.py

# Predict
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"temperature": 35, "substrate_volume": 100, "retention_time": 20}'
```

---

## 📦 Docker

```bash
docker-compose up -d
```

This starts:
- MLflow tracking server
- FastAPI inference API

---

## 📊 MLflow Tracking

All experiments are logged with:
- Parameters
- Metrics (RMSE, MAE, R²)
- Artifacts (model, plots)
- Registered model versions

---

## 🤝 Connect

Built by [Brunno Bachmann](https://www.linkedin.com/in/brunno-bachmann-865429173) as part of a portfolio transition into MLOps and Applied AI.

---

## 📄 License

MIT
