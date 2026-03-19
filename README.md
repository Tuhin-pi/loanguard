# 🛡️ LoanGuard — Production ML Platform

> An end-to-end loan default prediction system with automated data drift detection, model retraining pipeline, FastAPI inference, and Streamlit frontend.

![Python](https://img.shields.io/badge/Python-3.10-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange) ![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green) ![MLflow](https://img.shields.io/badge/MLflow-2.8-blue) ![Docker](https://img.shields.io/badge/Docker-Containerized-blue) ![Evidently](https://img.shields.io/badge/Evidently-Drift%20Detection-purple)

---

## 📌 Overview

LoanGuard is a production-grade machine learning platform that predicts whether a loan applicant is likely to default. It goes beyond just model training — it monitors incoming data for drift, automatically retrains the model when needed, and only promotes the new model if it outperforms the previous one.

**Model Performance**
- ROC-AUC: **0.75** on 61,000+ validation samples
- Trained on 246,000+ real loan applications (Home Credit dataset)
- Handles severe class imbalance (92% non-default vs 8% default)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        LoanGuard Platform                    │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │  Streamlit   │───▶│  FastAPI     │───▶│  XGBoost      │  │
│  │  Frontend    │    │  Inference   │    │  Model        │  │
│  │  :8501       │    │  API :8000   │    │               │  │
│  └──────────────┘    └──────┬───────┘    └───────────────┘  │
│                             │                                │
│                    Logs predictions to CSV                   │
│                             │                                │
│                    ┌────────▼────────┐                       │
│                    │  Evidently AI   │                       │
│                    │  Drift Detector │                       │
│                    └────────┬────────┘                       │
│                             │ Drift detected?                │
│                    ┌────────▼────────┐                       │
│                    │  Auto Retrain   │                       │
│                    │  + MLflow Log   │                       │
│                    └────────┬────────┘                       │
│                             │ AUC improved?                  │
│                    ┌────────▼────────┐                       │
│                    │ Model Promotion │                       │
│                    │  (best_model)   │                       │
│                    └─────────────────┘                       │
│                                                              │
│  ┌──────────────┐                                            │
│  │  MLflow UI   │  Experiment tracking & model registry      │
│  │  :5000       │                                            │
│  └──────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
loanguard/
├── data/
│   ├── raw/                        # Raw CSV data (not committed)
│   ├── processed/                  # Reference data for drift detection
│   ├── predictions/                # Logged predictions for monitoring
│   └── drift_reports/              # Evidently HTML + JSON reports
├── src/
│   ├── ingestion.py                # Data loading & validation
│   ├── preprocessing.py            # Feature engineering pipeline
│   ├── train.py                    # Model training + MLflow logging
│   ├── evaluate.py                 # Save best model from MLflow
│   ├── drift_detector.py           # Evidently AI drift detection
│   ├── retrain.py                  # Auto retraining + model promotion
│   └── api.py                      # FastAPI inference endpoint
│   └── app.py                      # Streamlit frontend
├── models/
│   └── best_model.json             # Active XGBoost model
├── tests/                          # Unit tests
├── mlruns/                         # MLflow experiment data (auto-generated)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Kaggle account (for dataset)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/loanguard.git
cd loanguard
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install streamlit
```

### 3. Download the Dataset

Download `application_train.csv` from the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) Kaggle competition and place it in:

```
data/raw/application_train.csv
```

### 4. Run the Full Pipeline

```bash
# Step 1 — Load and validate data
python src/ingestion.py

# Step 2 — Feature engineering + save reference data
python src/preprocessing.py

# Step 3 — Train model and log to MLflow
python src/train.py

# Step 4 — Save best model artifact
python src/evaluate.py
```

---

## 🐳 Running with Docker

Start the API and MLflow together:

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| FastAPI Inference API | http://localhost:8000 |
| FastAPI Swagger Docs | http://localhost:8000/docs |
| MLflow Dashboard | http://localhost:5000 |

To stop:

```bash
docker-compose down
```

---

## 🖥️ Running the Frontend

Make sure FastAPI is running first, then in a separate terminal:

```bash
streamlit run src/app.py
```

Open **http://localhost:8501** in your browser.

![LoanGuard Frontend](https://via.placeholder.com/800x400?text=LoanGuard+Streamlit+Frontend)

The frontend allows you to:
- Enter applicant financial and personal details
- Get real-time risk assessment (LOW / MEDIUM / HIGH)
- View default probability score
- See application summary table

---

## 🔍 API Reference

### Health Check

```bash
GET /health
```

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Predict Default Risk

```bash
POST /predict
```

**Request body:**

```json
{
  "AMT_INCOME_TOTAL": 135000,
  "AMT_CREDIT": 568800,
  "AMT_ANNUITY": 20560,
  "AMT_GOODS_PRICE": 450000,
  "DAYS_BIRTH": -12005,
  "DAYS_EMPLOYED": -2500,
  "DAYS_REGISTRATION": -3648.0,
  "DAYS_ID_PUBLISH": -2700,
  "CNT_FAM_MEMBERS": 2.0,
  "CNT_CHILDREN": 0,
  "EXT_SOURCE_1": 0.6,
  "EXT_SOURCE_2": 0.7,
  "EXT_SOURCE_3": 0.6,
  "CODE_GENDER": 1,
  "NAME_CONTRACT_TYPE": 0,
  "FLAG_OWN_CAR": 0,
  "FLAG_OWN_REALTY": 1
}
```

**Response:**

```json
{
  "default_probability": 0.1823,
  "risk_level": "LOW",
  "recommendation": "Approve — low default risk"
}
```

| Risk Level | Probability | Action |
|------------|-------------|--------|
| 🟢 LOW | < 30% | Approve |
| 🟡 MEDIUM | 30% – 60% | Manual review |
| 🔴 HIGH | > 60% | Reject |

---

## 📊 Drift Detection & Auto Retraining

LoanGuard monitors incoming predictions for data drift using **Evidently AI**. When significant drift is detected, it automatically triggers retraining and only promotes the new model if performance improves.

```bash
# Run drift detection + auto retraining pipeline
python src/retrain.py
```

**How it works:**

1. Incoming predictions are logged to `data/predictions/prediction_log.csv`
2. Evidently compares this data against the original training distribution
3. If more than **20% of features** drift significantly, retraining is triggered
4. A new XGBoost model is trained and evaluated
5. The new model replaces the old one **only if AUC improves**
6. All runs are logged to MLflow with a `promoted: true/false` tag

**Drift reports** are saved as HTML in `data/drift_reports/` for full transparency.

---

## 📈 MLflow Experiment Tracking

View all experiments, metrics, and model artifacts:

```bash
mlflow ui
# Open http://localhost:5000
```

Each run tracks:
- `roc_auc` — primary model performance metric
- `precision`, `recall`, `f1` — per-class metrics
- All hyperparameters
- `promoted` tag — whether the model was deployed

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Model | XGBoost |
| Experiment Tracking | MLflow |
| Drift Detection | Evidently AI |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containerization | Docker + Docker Compose |
| Data Processing | Pandas, NumPy, Scikit-learn |

---

## 🗺️ Roadmap

- [x] Data ingestion & validation pipeline
- [x] Feature engineering with domain-specific ratios
- [x] XGBoost training with MLflow tracking
- [x] FastAPI inference endpoint
- [x] Docker containerization
- [x] Evidently AI drift detection
- [x] Automated retraining + model promotion
- [x] Streamlit frontend
- [ ] AWS EC2 deployment
- [ ] GitHub Actions CI/CD pipeline
- [ ] Grafana monitoring dashboard

---

## 👤 Author

**Tuhin Maity** — AI/ML Engineer  
[GitHub](https://github.com/tuhin) · [LinkedIn](https://linkedin.com/in/tuhin) · tuhin3024@gmail.com