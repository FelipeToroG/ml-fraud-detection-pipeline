# ML Fraud Detection Pipeline

> **End-to-end production ML system for detecting fraudulent credit card transactions.** Built with sklearn Pipelines, gradient boosting (XGBoost, LightGBM, CatBoost), Optuna hyperparameter tuning, MLflow experiment tracking, SHAP explainability, and a FastAPI inference service deployed via Docker.

---

## TL;DR

A senior-level ML engineering project that takes a real-world fraud detection dataset (250,000 transactions, 0.18% fraud rate) and ships a production-ready classifier — not just a notebook with a confusion matrix.

| Layer | Stack |
|---|---|
| **Data** | sklearn ColumnTransformer + Pipeline (zero leakage) |
| **Modeling** | Logistic Regression · Random Forest · XGBoost · LightGBM · CatBoost · MLP |
| **Tuning** | Optuna (Bayesian optimization, 50–100 trials per model) |
| **Tracking** | MLflow (every run logged, comparable, reproducible) |
| **Explainability** | SHAP (global feature importance + per-prediction attribution) |
| **Threshold Selection** | Cost-sensitive optimization (not just F1) |
| **Serving** | FastAPI + Pydantic + Docker |
| **Testing** | pytest (feature pipelines + API contracts) |

---

## Why This Project Exists

Fraud detection is one of the most studied ML problems on the planet. What makes a portfolio version stand out isn't the model accuracy — it's whether the work demonstrates the rigor of a production ML engineer:

- **No data leakage** — feature transformations live inside the pipeline, fit on training only
- **Cost-sensitive evaluation** — false negatives (missed fraud) cost more than false positives, and the threshold reflects that
- **Tracked experiments** — every model run is reproducible via MLflow
- **Explainability** — SHAP values for every prediction, not just feature importance
- **Deployable** — actually runs as a service, not just a Jupyter cell

This README will be filled out as the project completes. See [Project Status](#project-status) below.

---

## Project Structure

```
ml-fraud-detection-pipeline/
├── data/
│   ├── raw/              # Original immutable data
│   ├── processed/        # Cleaned, feature-engineered data
│   └── external/         # External reference data
│
├── notebooks/            # Narrated, publication-quality analysis
│   ├── 01_eda.ipynb      # Exploratory data analysis
│   ├── 02_modeling.ipynb # Model training & evaluation
│   └── 03_explainability.ipynb # SHAP analysis
│
├── src/
│   ├── data/             # Data loading & validation
│   ├── features/         # Feature engineering pipelines
│   ├── models/           # Training, tuning, evaluation
│   ├── api/              # FastAPI inference service
│   └── utils/            # Shared utilities
│
├── tests/                # pytest unit & integration tests
├── mlruns/               # MLflow experiment artifacts
├── configs/              # YAML config files
├── docs/                 # Architecture diagrams, results
│
├── Dockerfile            # Production deployment image
├── requirements.txt      # Pinned dependencies
└── README.md             # This file
```

---

## Quickstart

```bash
# Clone and set up
git clone https://github.com/FelipeToroG/ml-fraud-detection-pipeline.git
cd ml-fraud-detection-pipeline

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run EDA
jupyter notebook notebooks/01_eda.ipynb

# Train models (Session 2)
python -m src.models.train

# Serve predictions (Session 3)
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```

---

## Dataset

**Source:** Anonymized credit card transactions with PCA-transformed features for confidentiality.

| Property | Value |
|---|---|
| Rows | 249,999 |
| Features | 30 (X1–X28 PCA components, X29 transaction amount, X30 time) |
| Target | `Response` (1 = fraud, 0 = legitimate) |
| Class imbalance | 0.183% fraud (458 of 249,999) |

The severe class imbalance is the central modeling challenge. The dominant cost in fraud detection is the **missed fraud (false negative)** — a single missed transaction can cost thousands of dollars while a flagged legitimate transaction costs only review time. This drives every modeling decision in the project.

---

## Results

> _Populated in Session 2._ Will include:
> - Full results table (all 6 models, with CV scores)
> - ROC curves and Precision-Recall curves overlaid
> - SHAP summary plots
> - Cost-sensitive threshold analysis

---

## Architecture

> _Populated in Session 3._ Will include a system diagram showing data flow from training through inference.

---

## Project Status

| Session | Scope | Status |
|---|---|---|
| **1** | Project structure + EDA notebook + cost framing | ✅ Complete |
| **2** | Pipeline + 6 models + Optuna + MLflow + SHAP | ⏳ In Progress |
| **3** | FastAPI service + Dockerfile + tests + final README | ⏳ Pending |

---

## Author

**Felipe Toro** — AI/ML Engineer transitioning from BI Engineering · MS in AI & Business Analytics @ University of South Florida

🔗 [Portfolio](https://felipetorog.github.io/Portfolio) · [LinkedIn](https://www.linkedin.com/in/felipe-toro-g/) · [GitHub](https://github.com/FelipeToroG)

---

## License

MIT — see `LICENSE` file.
