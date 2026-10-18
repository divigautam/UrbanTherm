from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
import os
import re
import json
import math
import urllib.parse
import urllib.request
from functools import lru_cache

import numpy as np

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
# ADDRESS SEARCH SETUP
# (typed area/address -> coordinates -> nearest grid cell)
# =========================

GEOCODE_URL = "https://nominatim.openstreetmap.org/search"
GEOCODE_USER_AGENT = "UrbanTherm/1.0 (academic project)"

# Cells are 500 m x 500 m, so a point inside a cell is at most ~354 m
# from that cell's centre. Anything further is outside the grid.
MAX_MATCH_DISTANCE_M = 400
BBOX_PAD_DEG = 0.02

_grid_lat = np.radians(data["centroid_lat"].to_numpy(dtype=float))
_grid_lon = np.radians(data["centroid_lon"].to_numpy(dtype=float))

# Limit geocoding to the area your grid covers (taken from the data itself)
# Nominatim viewbox order: left(lon), top(lat), right(lon), bottom(lat)
_VIEWBOX = "{},{},{},{}".format(
    data["centroid_lon"].min() - BBOX_PAD_DEG,
    data["centroid_lat"].max() + BBOX_PAD_DEG,
    data["centroid_lon"].max() + BBOX_PAD_DEG,
    data["centroid_lat"].min() - BBOX_PAD_DEG,
)


@lru_cache(maxsize=256)
def geocode(query):
    """Return (lat, lon, display_name) for a typed place, or None."""

    params = urllib.parse.urlencode({
        "q": query,
        "format": "jsonv2",
        "limit": 1,
        "bounded": 1,
        "viewbox": _VIEWBOX,
        "countrycodes": "in"
    })

    request_obj = urllib.request.Request(
        f"{GEOCODE_URL}?{params}",
        headers={"User-Agent": GEOCODE_USER_AGENT}
    )

    with urllib.request.urlopen(request_obj, timeout=8) as response:
        results = json.loads(response.read().decode("utf-8"))

    if not results:
        return None

    best = results[0]

    return float(best["lat"]), float(best["lon"]), best.get("display_name", query)


def nearest_grid_index(lat, lon):
    """Index of the closest grid centroid and its distance in metres."""

    lat_r = math.radians(lat)
    lon_r = math.radians(lon)

    a = (
        np.sin((_grid_lat - lat_r) / 2) ** 2
        + np.cos(lat_r) * np.cos(_grid_lat) * np.sin((_grid_lon - lon_r) / 2) ** 2
    )

    distances = 2 * 6371000 * np.arcsin(np.sqrt(a))
    idx = int(distances.argmin())

    return idx, float(distances[idx])


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


@app.route("/api/predict-address", methods=["GET"])
def predict_address():

    query = request.args.get("q", "").strip()

    if not query:
        return jsonify({
            "error": "Please enter an area or address"
        }), 400

    # A grid ID such as G1086 still works exactly as before
    if re.fullmatch(r"G\d{1,5}", query, re.IGNORECASE):
        return predict_grid(query)

    try:
        place = geocode(query.lower())
    except Exception:
        return jsonify({
            "error": "Address lookup service is unavailable, please try again"
        }), 502

    if place is None:
        return jsonify({
            "error": "Could not find that place inside the covered area. "
                     "Try a locality or landmark name."
        }), 404

    lat, lon, display_name = place

    idx, distance_m = nearest_grid_index(lat, lon)

    if distance_m > MAX_MATCH_DISTANCE_M:
        return jsonify({
            "error": "That place is outside the UrbanTherm grid coverage"
        }), 404

    grid_id = str(data.iloc[idx]["grid_id"])

    # Reuse the existing route so the response stays identical
    payload = predict_grid(grid_id).get_json()

    payload.update({
        "query": query,
        "matched_address": display_name,
        "searched_lat": lat,
        "searched_lon": lon,
        "distance_to_cell_centre_m": round(distance_m, 1)
    })

    return jsonify(payload)



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