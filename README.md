UrbanTherm 🌡️🏙️

AI-Powered Hyperlocal Urban Heat Intelligence for Ghaziabad, Uttar Pradesh

UrbanTherm is a climate-tech web application designed to explore and analyze urban heat patterns at a fine spatial scale. It combines satellite-derived land surface temperature, vegetation and built-up indicators, historical weather features, spatial grid analysis, and a Random Forest machine-learning model to provide grid-level heat-risk intelligence for Ghaziabad.

The project transforms environmental observations into an interactive interface where users can explore heat-risk maps, inspect individual urban cells, and view machine-learning predictions.

📌 Project Overview

Urban heat is not distributed uniformly across a city. Built-up surfaces, vegetation, and local environmental conditions can create significant differences in thermal conditions between nearby locations.

UrbanTherm addresses this problem by dividing the Ghaziabad study area into 500m × 500m spatial grid cells and associating environmental features with each cell.

The application provides:

Interactive heat-risk maps

Grid-level environmental intelligence

Satellite-derived LST, NDVI and NDBI

Weather-derived environmental features

Machine-learning heat-risk classification

Flask REST API

Premium climate-tech web interface

🎯 Objectives

Analyze spatial variation in urban heat across Ghaziabad.

Generate a fine-resolution spatial grid for localized analysis.

Extract satellite-derived environmental indicators.

Combine satellite and weather information into a unified dataset.

Train a machine-learning model for heat-risk classification.

Provide an interactive web interface for exploring the results.

Enable users to inspect individual grid cells through the UrbanTherm API.

✨ Key Features

🗺️ Heat Map

Visualizes the spatial distribution of observed heat-risk categories across Ghaziabad.

🔎 Grid Intelligence

Users can enter a grid ID such as G1086 and retrieve its environmental and heat-risk information.

🤖 AI Prediction

Uses a trained Random Forest classifier to generate a heat-risk prediction for a selected grid cell.

🛰️ Satellite Intelligence

Uses Landsat-derived:

Land Surface Temperature (LST)

NDVI — Normalized Difference Vegetation Index

NDBI — Normalized Difference Built-up Index

🌦️ Weather Features

Historical weather data contributes:

Temperature

Relative humidity

Wind speed

Heat-index-derived feature

🔐 Login Interface

A demonstration login flow provides controlled access to the application interface.

🧠 System Architecture

                 ┌──────────────────────┐
                 │   Ghaziabad Study    │
                 │        Area          │
                 └──────────┬───────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
     ┌────────────────┐         ┌────────────────┐
     │ Landsat Data   │         │ Weather Data   │
     │                │         │                │
     │ Red / NIR      │         │ Temperature    │
     │ SWIR / Thermal │         │ Humidity       │
     └───────┬────────┘         │ Wind Speed     │
             │                  └───────┬────────┘
             ▼                          ▼
     ┌────────────────┐         ┌────────────────┐
     │ NDVI / NDBI /  │         │ Weather        │
     │ LST Features   │         │ Features       │
     └───────┬────────┘         └───────┬────────┘
             │                          │
             └─────────────┬────────────┘
                           ▼
                 ┌────────────────────┐
                 │ Feature Integration │
                 └──────────┬─────────┘
                            ▼
                 ┌────────────────────┐
                 │ 500m Spatial Grid  │
                 │    3,881 cells     │
                 └──────────┬─────────┘
                            ▼
                 ┌────────────────────┐
                 │ Random Forest ML   │
                 │    Classifier      │
                 └──────────┬─────────┘
                            ▼
                 ┌────────────────────┐
                 │ Flask REST API     │
                 └──────────┬─────────┘
                            ▼
          ┌─────────────────────────────────┐
          │       UrbanTherm Frontend       │
          │                                 │
          │ Dashboard · Heat Map            │
          │ AI Prediction · Grid Intelligence│
          │ About                           │
          └─────────────────────────────────┘

🛰️ Data Sources

Satellite Data

Landsat Collection 2 Level-2 data was accessed through the Microsoft Planetary Computer STAC API.

Relevant bands/assets:

Red

NIR (nir08)

SWIR (swir16)

Thermal (lwir11)

Satellite-derived features include:

NDVI

NDVI = (NIR - Red) / (NIR + Red)

NDBI

NDBI = (SWIR - NIR) / (SWIR + NIR)

Land Surface Temperature

The thermal band was converted using the Landsat scaling parameters and expressed in Celsius.

🌦️ Weather Data

Historical hourly weather data for 2025 was obtained using the Open-Meteo historical weather API.

The dataset contains:

Temperature

Relative humidity

Wind speed

A simple derived heat-index feature was calculated during feature engineering:

heat_index_simple = temperature + 0.05 × humidity

Daily weather features were then integrated with the spatial satellite features.

🗺️ Spatial Grid

The Ghaziabad study area was converted into a regular spatial grid.

Property

Value

Study Area

Ghaziabad, Uttar Pradesh

Grid Size

500m × 500m

Number of Cells

3,881

Projection

EPSG:32643

Maximum Cell Area

250,000 m²

The grid allows UrbanTherm to analyze heat conditions at the individual urban-cell level instead of only using city-wide averages.

🤖 Machine Learning

UrbanTherm uses a Random Forest Classifier.

Input Features

NDVI
NDBI
Average Temperature
Maximum Temperature
Average Humidity
Average Wind Speed
Average Heat Index

Model Configuration

Algorithm: Random Forest Classifier
Estimators: 200
Random State: 42

Current Prototype Performance

The current model achieved approximately:

Accuracy: 57.53%

Class-level performance:

Risk Class

Precision

Recall

F1-score

High

0.61

0.69

0.64

Low

0.66

0.67

0.67

Medium

0.43

0.36

0.39

Feature Importance

The current trained model shows the strongest contribution from:

NDVI ≈ 0.509
NDBI ≈ 0.491

The weather features have zero importance in the current trained prototype because the available training target and data structure are dominated by the spatial satellite-derived signal.

📊 Current Observed Risk Distribution

For the current satellite observation used in the prototype:

Risk Category

Grid Cells

High

1,320

Medium

1,280

Low

1,281

Total

3,881

The observed land-surface-temperature range in the processed satellite dataset is approximately:

Minimum: 31.50 °C
Maximum: 54.96 °C
Mean:    43.75 °C

🧪 Important Scientific Limitation

UrbanTherm is currently a prototype spatial heat-intelligence system, not a validated operational heat forecast system.

The current satellite heat-risk analysis is based on one Landsat observation date. Therefore, the present result should be interpreted as an observed spatial heat-risk snapshot rather than a validated future-temperature forecast.

The machine-learning model is also a prototype classifier trained using the available spatial dataset.

For a stronger forecasting system, future versions should use:

Multiple satellite observation dates

Larger multi-season datasets

More years of weather observations

Ground-based temperature measurements

Temporal train/test validation

Additional urban morphology features

Model comparison and hyperparameter optimization

🛠️ Technology Stack

Frontend

HTML5

CSS3

JavaScript

Fetch API

Folium-generated map integration

Backend

Python

Flask

Flask-CORS

Data Science

Pandas

NumPy

Scikit-learn

Joblib

Geospatial

GeoPandas

OSMnx

Rasterio

Shapely

PySTAC Client

Microsoft Planetary Computer

Data Sources

Open-Meteo

Landsat Collection 2

OpenStreetMap

📁 Project Structure

UrbanTherm/
│
├── backend/
│   └── app.py
│
├── frontend/
│   ├── index.html
│   ├── login.html
│   ├── dashboard.html
│   ├── heatmap.html
│   ├── prediction.html
│   ├── grid.html
│   ├── about.html
│   └── motion.css
│
├── data/
│   ├── raw/
│   │   └── ...
│   │
│   └── processed/
│       ├── ghaziabad_grid_500m.geojson
│       ├── ghaziabad_grid_500m.csv
│       ├── ghaziabad_weather.csv
│       ├── ghaziabad_weather_features.csv
│       ├── ghaziabad_satellite_features.csv
│       ├── ghaziabad_final_features.csv
│       └── ghaziabad_real_heat_risk.csv
│
├── docs/
│   ├── ghaziabad_grid_preview.png
│   ├── ghaziabad_heat_risk_map.html
│   ├── ghaziabad_high_risk_map.html
│   ├── ghaziabad_medium_risk_map.html
│   ├── ghaziabad_low_risk_map.html
│   └── heat_risk_trend.png
│
├── ml/
│   └── heat_risk_model.pkl
│
├── scripts/
│   ├── fetch_weather.py
│   ├── feature_engineering.py
│   ├── create_heat_risk.py
│   ├── grid_satellite_features.py
│   ├── merge_weather_satellite.py
│   ├── create_real_heat_risk.py
│   └── visualize_heat_risk.py
│
├── .gitignore
└── README.md

🚀 Setup

1. Clone the repository

git clone https://github.com/divigautam/UrbanTherm.git
cd UrbanTherm

2. Install Python dependencies

python -m pip install pandas numpy scikit-learn joblib flask flask-cors
python -m pip install geopandas osmnx shapely rasterio pystac-client planetary-computer

3. Start the Flask backend

From the project root:

python backend/app.py

The API will run at:

http://127.0.0.1:5000

4. Open the frontend

Open:

frontend/index.html

in a browser.

🔌 API Endpoints

Health Check

GET /

Retrieve Grid Dataset

GET /api/grid

Predict a Specific Grid

GET /api/predict-grid/<grid_id>

Example:

http://127.0.0.1:5000/api/predict-grid/G1086

Custom Prediction

POST /api/predict

The API accepts the model's environmental feature values and returns a predicted heat-risk class.

🖥️ Application Pages

Page

Purpose

index.html

UrbanTherm landing page

login.html

Demo authentication

dashboard.html

Overall heat-risk dashboard

heatmap.html

Spatial heat-risk maps

prediction.html

AI grid prediction

grid.html

Detailed grid intelligence

about.html

Project methodology and information

Application flow:

Landing
   ↓
Login
   ↓
Dashboard
   ├── Heat Map
   ├── AI Prediction
   ├── Grid Intelligence
   └── About

🔐 Demo Login

The current login page uses demonstration authentication for the college project interface.

Email:    admin@urbanthem.com
Password: admin123

This is not production-grade authentication and does not represent a real user-management system.

🔮 Future Scope

UrbanTherm can be extended into a more robust urban heat forecasting platform by adding:

Multi-temporal satellite imagery

Real-time weather feeds

IoT temperature sensors

Heatwave forecasting

Temporal deep-learning models

XGBoost / LightGBM model comparison

Explainable AI

Population exposure analysis

Vulnerability mapping

Heat-health alerts

Ward-level planning support

Mobile-friendly emergency heat notifications

Cloud deployment and scalable APIs

🎓 Academic Value

UrbanTherm combines multiple areas of computer science and geospatial technology:

Web Development
        +
REST API Development
        +
Machine Learning
        +
Data Engineering
        +
GIS / Spatial Analysis
        +
Remote Sensing
        =
UrbanTherm

This makes the project suitable as a research + application-oriented academic project rather than a simple static web application.

⚠️ Disclaimer

UrbanTherm is an academic prototype developed for research, demonstration and educational purposes.

Its current heat-risk outputs should not be treated as official public-safety warnings or medical guidance. The present model and satellite analysis require further validation before operational deployment.

👩‍💻 Project

UrbanTherm — AI-Powered Hyperlocal Urban Heat Intelligence

Study Area: Ghaziabad, Uttar Pradesh, India

Repository:
https://github.com/divigautam/UrbanTherm

<p align="center">
  Built with 🛰️ satellite data, 🌡️ climate intelligence and 🤖 machine learning.
</p>