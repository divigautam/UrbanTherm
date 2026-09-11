from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)

# =========================
# PROJECT PATHS
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
MODEL_PATH = os.path.join(BASE_DIR, "ml", "heat_risk_model.pkl")
DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ghaziabad_real_heat_risk.csv"
)

# =========================
# LOAD MODEL + DATA
# =========================

model = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH)


# =========================
# API ROUTES
# =========================

@app.route("/api/grid", methods=["GET"])
def get_grid():

    result = data[
        [
            "grid_id",
            "centroid_lat",
            "centroid_lon",
            "ndvi",
            "ndbi",
            "lst_celsius",
            "heat_risk",
            "heat_risk_score",
            "avg_temperature",
            "max_temperature",
            "avg_humidity",
            "avg_wind_speed",
            "avg_heat_index"
        ]
    ]

    return jsonify(result.to_dict(orient="records"))


@app.route("/api/predict-grid/<grid_id>", methods=["GET"])
def predict_grid(grid_id):

    grid = data[
        data["grid_id"].astype(str).str.upper()
        == grid_id.upper()
    ]

    if grid.empty:
        return jsonify({
            "error": "Grid not found"
        }), 404

    row = grid.iloc[0]

    features = [[
        row["ndvi"],
        row["ndbi"],
        row["avg_temperature"],
        row["max_temperature"],
        row["avg_humidity"],
        row["avg_wind_speed"],
        row["avg_heat_index"]
    ]]

    prediction = model.predict(features)[0]

    return jsonify({
        "grid_id": row["grid_id"],
        "ml_prediction": prediction,
        "observed_risk": row["heat_risk"],
        "lst_celsius": row["lst_celsius"],
        "ndvi": row["ndvi"],
        "ndbi": row["ndbi"],
        "heat_risk_score": row["heat_risk_score"],
        "avg_temperature": row["avg_temperature"],
        "max_temperature": row["max_temperature"],
        "avg_humidity": row["avg_humidity"],
        "avg_wind_speed": row["avg_wind_speed"],
        "avg_heat_index": row["avg_heat_index"]
    })


@app.route("/api/predict", methods=["POST"])
def predict():

    values = request.json

    features = [[
        values["ndvi"],
        values["ndbi"],
        values["avg_temperature"],
        values["max_temperature"],
        values["avg_humidity"],
        values["avg_wind_speed"],
        values["avg_heat_index"]
    ]]

    prediction = model.predict(features)[0]

    return jsonify({
        "predicted_heat_risk": prediction
    })


# =========================
# HEALTH CHECK
# =========================

@app.route("/health")
def health():

    return jsonify({
        "status": "healthy",
        "service": "UrbanTherm API"
    })


# =========================
# MAP / DOCS
# =========================

@app.route("/docs/<path:filename>")
def serve_docs(filename):

    return send_from_directory(
        DOCS_DIR,
        filename
    )


# =========================
# FRONTEND
# =========================

@app.route("/")
def home():

    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


@app.route("/<path:path>")
def serve_frontend(path):

    file_path = os.path.join(
        FRONTEND_DIR,
        path
    )

    if os.path.isfile(file_path):

        return send_from_directory(
            FRONTEND_DIR,
            path
        )

    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


# =========================
# LOCAL RUN
# =========================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )