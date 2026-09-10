from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)


# ==============================
# LOAD MODEL
# ==============================

model = joblib.load("ml/heat_risk_model.pkl")


# ==============================
# LOAD DATA
# ==============================

data = pd.read_csv(
    "data/processed/ghaziabad_real_heat_risk.csv"
)


# ==============================
# HOME
# ==============================

@app.route("/")
def home():

    return jsonify({
        "message": "UrbanTherm API is running!"
    })


# ==============================
# ALL GRID DATA
# ==============================

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

    return jsonify(
        result.to_dict(orient="records")
    )


# ==============================
# SINGLE GRID + ML PREDICTION
# ==============================

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


    # Features used by Random Forest
    features = [[
        row["ndvi"],
        row["ndbi"],
        row["avg_temperature"],
        row["max_temperature"],
        row["avg_humidity"],
        row["avg_wind_speed"],
        row["avg_heat_index"]
    ]]


    # ML prediction
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


# ==============================
# MANUAL ML PREDICTION
# ==============================

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


# ==============================
# START SERVER
# ==============================

if __name__ == "__main__":

    app.run(
        debug=True
    )