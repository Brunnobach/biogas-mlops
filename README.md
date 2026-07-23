# 🌱 Biogas MLOps

**End-to-end MLOps pipeline for biogas production forecasting**

A production-ready machine learning system that predicts daily biogas production from operational parameters, with MLflow experiment tracking, model registry, a containerized FastAPI inference service and automated CI/CD.

---

## Live portfolio & predictor UI

**Interactive predictor:** https://brunnobach.github.io/biogas-mlops/

The GitHub Pages app is a usable forecasting tool (Portuguese UI). On Pages it runs a client-side physics-inspired formula (demo mode). With the local FastAPI service running, click **Conectar** and predictions use the trained model via `POST /predict` (CORS enabled).

---

## 🎯 What this project demonstrates

| Skill | How it is applied here |
|-------|------------------------|
| **MLOps** | MLflow tracking, model registry, reproducible training pipeline |
| **ML Engineering** | Feature engineering, Gradient Boosting model, metrics evaluation |
| **Cloud-ready ML** | Dockerized FastAPI inference API |
| **CI/CD** | GitHub Actions for testing, training and Docker builds |
| **Domain expertise** | Biogas production forecasting with real-world parameters |
| **Software engineering** | Modular code, unit tests, logging, environment configuration |

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
                                              │  FastAPI API      │
                                              │  /predict         │
                                              │  (Docker)         │
                                              └─────────────────┘
```

---

## 🛠️ Tech Stack

- **Python 3.11**
- **scikit-learn** (GradientBoostingRegressor)
- **MLflow** (experiment tracking and model registry)
- **FastAPI** (inference API)
- **Docker / Docker Compose**
- **pytest** (unit tests)
- **GitHub Actions** (CI/CD)

---

## 📁 Project Structure

```
biogas-mlops/
├── .github/workflows/      # CI/CD pipelines
├── data/                   # Datasets
├── models/                 # Trained models and metrics
├── notebooks/              # EDA and experiments
├── src/
│   ├── data/               # Dataset generation
│   ├── features/           # Feature engineering
│   ├── models/             # Training script
│   ├── api/                # FastAPI app
│   └── utils/              # Configuration
├── tests/                  # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start (Local)

```bash
git clone https://github.com/Brunnobach/biogas-mlops.git
cd biogas-mlops

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate data
python src/data/generate_dataset.py

# Train model with MLflow tracking
python src/models/train.py

# Run API locally
python -m uvicorn src.api.app:app --reload
```

Test the API:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 37.0,
    "ph": 7.2,
    "organic_load": 3.5,
    "hydraulic_retention_time": 25,
    "substrate_type": "cattle_swine_poultry",
    "humidity": 85,
    "ambient_temperature": 22,
    "previous_day_production": 120,
    "month": 6
  }'
```

---

## 🐳 Docker Compose (Full Stack)

```bash
docker-compose up -d
```

This starts:
- **MLflow** tracking server at http://localhost:5000
- **FastAPI** inference service at http://localhost:8000

The API container automatically generates data, trains the model and starts serving predictions.

---

## 📊 MLflow Tracking

All training runs are logged with:
- Hyperparameters (n_estimators, learning_rate, max_depth)
- Metrics (RMSE, MAE, R²)
- Model artifact (scikit-learn pipeline)
- Local model file

Access MLflow UI: http://localhost:5000

---

## 🧪 Dataset

The synthetic dataset includes operational parameters inspired by real biodigesters:

| Feature | Description |
|---------|-------------|
| `temperature` | Digester temperature (°C) |
| `ph` | Digester pH |
| `organic_load` | Organic load (kg VS / m³ / day) |
| `hydraulic_retention_time` | HRT (days) |
| `substrate_type` | Type of organic substrate |
| `humidity` | Substrate humidity (%) |
| `ambient_temperature` | External temperature (°C) |
| `previous_day_production` | Lag feature (m³ / day) |
| `biogas_production` | **Target** (m³ / day) |

---

## ✅ CI/CD

Every push to `main` triggers:
1. Dependency installation
2. Data generation and model training
3. Unit tests
4. Docker image build
5. GitHub Pages deployment

---

## 🤝 Connect

Built by [Brunno Bachmann](https://www.linkedin.com/in/brunno-bachmann-865429173) as part of a portfolio transition into MLOps and Applied AI.

If you are interested in biogas, sustainability, MLOps or industrial digitalization, feel free to connect.

---

## 📄 License

MIT
