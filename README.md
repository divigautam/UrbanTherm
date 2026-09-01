# UrbanTherm

UrbanTherm is an AI-powered hyperlocal urban heat intelligence and risk forecasting system. This repository currently contains only the **initial project structure** and a minimal FastAPI backend. Data collection, ML models, and the full frontend are not built yet.

## Folder purpose

| Folder | Purpose |
| --- | --- |
| `backend/` | FastAPI web API. Serves heat-risk results to the frontend later. |
| `backend/app/` | Python package with the API application code. |
| `frontend/` | React + Vite + Tailwind + Leaflet map UI (not built yet). |
| `ml/` | Machine learning code (XGBoost, SHAP, scikit-learn) in later steps. |
| `ml/src/` | Reusable Python modules for training and inference. |
| `ml/notebooks/` | Jupyter notebooks for exploration and experiments. |
| `data/raw/` | Original downloaded datasets (satellite, weather, census). Keep this folder empty of fake data. |
| `data/processed/` | Cleaned features ready for modeling. Do not invent sample predictions. |
| `scripts/` | One-off helper scripts (download, preprocess, run jobs). |
| `docs/` | Project notes, architecture, and college report material. |
| `tests/` | Automated tests for API and later ML utilities. |

## Planned technology stack

- **Backend:** Python, FastAPI
- **ML:** Python, XGBoost, SHAP, Pandas, NumPy, Scikit-learn
- **Geospatial:** GeoPandas, Rasterio, Google Earth Engine
- **Database:** PostgreSQL, PostGIS
- **Frontend:** React, Vite, Tailwind CSS, Leaflet

## Setup (backend only)

Use Python 3.10 or newer. From the project root:

```bash
python -m venv .venv
```

Activate the virtual environment:

- Windows (PowerShell): `.\.venv\Scripts\Activate.ps1`
- macOS / Linux: `source .venv/bin/activate`

Install only what is needed to run the API:

```bash
pip install fastapi uvicorn
```

Copy environment placeholders (optional for this step):

```bash
copy .env.example .env
```

On macOS / Linux: `cp .env.example .env`

## Run the API

From the `backend` folder:

```bash
uvicorn app.main:app --reload
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000). You should see:

```json
{"message": "UrbanTherm API is running"}
```

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## What is not included yet

- No datasets (real or fake)
- No ML model or predictions
- No complete frontend
- No API keys or passwords
