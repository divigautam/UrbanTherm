from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
import os
import re
import json
import math
import ssl
import logging
import urllib.parse
import urllib.request
from functools import lru_cache

import numpy as np

try:
    import insights
except ImportError:  # started as a package, e.g. gunicorn backend.app:app
    from backend import insights

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

# Use certifi's CA bundle if available (avoids SSL errors on some Windows/Mac setups)
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = None

GEOCODE_URL = "https://nominatim.openstreetmap.org/search"
PHOTON_URL = "https://photon.komoot.io/api/"

# Nominatim blocks generic user agents. Put your real email here.
GEOCODE_USER_AGENT = (
    "UrbanTherm/1.0 (https://github.com/divigautam/UrbanTherm; ytworks2130@gmail.com)"
)

# Grid cells are 500 m x 500 m, so the centre of the cell containing a point
# is at most ~354 m away. Anything further than this is outside the grid.
MAX_MATCH_DISTANCE_M = 400
BBOX_PAD_DEG = 0.02

_grid_lat = np.radians(data["centroid_lat"].to_numpy(dtype=float))
_grid_lon = np.radians(data["centroid_lon"].to_numpy(dtype=float))

# Restrict geocoding to the area the grid covers (taken from the data itself)
_MIN_LON = float(data["centroid_lon"].min() - BBOX_PAD_DEG)
_MAX_LON = float(data["centroid_lon"].max() + BBOX_PAD_DEG)
_MIN_LAT = float(data["centroid_lat"].min() - BBOX_PAD_DEG)
_MAX_LAT = float(data["centroid_lat"].max() + BBOX_PAD_DEG)

# Nominatim viewbox order: left(lon), top(lat), right(lon), bottom(lat)
_VIEWBOX = f"{_MIN_LON},{_MAX_LAT},{_MAX_LON},{_MIN_LAT}"


def _get_json(url):
    """GET a URL and parse the JSON response."""

    req = urllib.request.Request(
        url,
        headers={"User-Agent": GEOCODE_USER_AGENT}
    )

    with urllib.request.urlopen(req, timeout=10, context=_SSL_CTX) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _geocode_nominatim(query):
    """Primary geocoder: OpenStreetMap Nominatim."""

    params = urllib.parse.urlencode({
        "q": query,
        "format": "jsonv2",
        "limit": 1,
        "bounded": 1,
        "viewbox": _VIEWBOX,
        "countrycodes": "in"
    })

    results = _get_json(f"{GEOCODE_URL}?{params}")

    if not results:
        return None

    best = results[0]

    return float(best["lat"]), float(best["lon"]), best.get("display_name", query)


def _geocode_photon(query):
    """Fallback geocoder: Photon (komoot), also OpenStreetMap based."""

    params = urllib.parse.urlencode({
        "q": query,
        "limit": 1,
        "bbox": f"{_MIN_LON},{_MIN_LAT},{_MAX_LON},{_MAX_LAT}"
    })

    results = _get_json(f"{PHOTON_URL}?{params}")

    features = results.get("features", [])

    if not features:
        return None

    lon, lat = features[0]["geometry"]["coordinates"]
    props = features[0].get("properties", {})

    name = ", ".join(
        str(props[key])
        for key in ("name", "district", "city", "state")
        if props.get(key)
    )

    return float(lat), float(lon), name or query


@lru_cache(maxsize=256)
def geocode(query):
    """
    Return (lat, lon, display_name) for a typed place, or None if not found.
    Tries Nominatim first, then Photon. Raises only if every provider errored.
    """

    last_error = None

    for provider in (_geocode_nominatim, _geocode_photon):
        try:
            result = provider(query)

            if result is not None:
                return result

        except Exception as e:
            last_error = e
            logging.exception("Geocoder %s failed", provider.__name__)

    if last_error is not None:
        raise last_error

    return None


def nearest_grid_index(lat, lon):
    """Index of the grid centroid closest to (lat, lon) and its distance in metres."""

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
    except Exception as e:
        app.logger.exception("Geocoding failed for %r", query)
        return jsonify({
            "error": f"Address lookup service is unavailable ({type(e).__name__}), please try again"
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


# =========================
# HEAT INSIGHTS (explanation)
# =========================

def _unpack(result):
    """A Flask view may return a Response or a (Response, status) tuple."""
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, result.status_code


def _insights_response(payload):
    grid_id = str(payload["grid_id"]).upper()

    row = data[
        data["grid_id"].astype(str).str.upper() == grid_id
    ].iloc[0]

    result = insights.explain_grid(
        row,
        data,
        payload.get("ml_prediction")
    )

    # keep the address-search details when the search came from an address
    for key in (
        "query",
        "matched_address",
        "searched_lat",
        "searched_lon",
        "distance_to_cell_centre_m"
    ):
        if key in payload:
            result[key] = payload[key]

    return jsonify(result)


@app.route("/api/insights", methods=["GET"])
def insights_by_query():

    # accepts ?q=<area or address or grid ID>, same as /api/predict-address
    response, status = _unpack(predict_address())

    if status != 200:
        return response, status

    return _insights_response(response.get_json())


@app.route("/api/insights/<grid_id>", methods=["GET"])
def insights_by_grid(grid_id):

    response, status = _unpack(predict_grid(grid_id))

    if status != 200:
        return response, status

    return _insights_response(response.get_json())


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
