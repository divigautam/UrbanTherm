"""Heat Insights: explain why a grid cell may have elevated (or lower) heat risk.

Rule-based and evidence-labelled. It uses only columns that already exist in the
UrbanTherm dataset (lst_celsius, ndvi, ndbi, heat_risk, heat_risk_score, date,
centroid_lat/lon and the weather columns).

Rules this module follows:
  * Every statement is labelled: measured, model_prediction,
    possible_explanation, context or unavailable.
  * Nothing here is live data.
  * NDVI / NDBI are indirect signals. Wording says "consistent with", never
    that a factor definitely caused the heat.
"""

import math

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- settings
HIGH_PCT = 67           # percentile at or above this -> "high" for the study area
LOW_PCT = 33            # percentile at or below this -> "low"
NEIGHBOUR_RADIUS_M = 800
NEIGHBOUR_DIFF_C = 1.5  # heuristic: "clearly different from surroundings"
MIN_NEIGHBOURS = 3
EARTH_RADIUS_M = 6371000

KIND_MEASURED = "measured"
KIND_MODEL = "model_prediction"
KIND_POSSIBLE = "possible_explanation"
KIND_CONTEXT = "context"
KIND_UNAVAILABLE = "unavailable"

WEATHER_COLUMNS = [
    "avg_temperature", "max_temperature", "avg_humidity", "avg_wind_speed",
]

INDICATORS = {
    "lst": {"column": "lst_celsius", "label": "Land surface temperature", "unit": "°C", "digits": 1},
    "ndvi": {"column": "ndvi", "label": "Vegetation index (NDVI)", "unit": "", "digits": 3},
    "ndbi": {"column": "ndbi", "label": "Built-up index (NDBI)", "unit": "", "digits": 3},
}

LIMITATIONS = [
    "Land surface temperature is the temperature of surfaces seen by the satellite "
    "on one date and time of day. It is not the air temperature people feel.",
    "NDVI and NDBI are indirect signals. They can suggest what may be heating or "
    "cooling an area, but they cannot confirm it.",
    "The ML model is a secondary check with limited accuracy (the project README "
    "reports about 57%). The observed satellite values are the main evidence.",
    "Results describe a 500 m grid cell, not a specific building or street.",
]

NOT_AVAILABLE = [
    "Tree canopy, shade and building density are not in the dataset.",
    "Land use and impervious-surface data are not in the dataset.",
    "Live weather is not part of this explanation.",
    "No estimate of how much any change would reduce temperature is available.",
]


# ----------------------------------------------------------------- helpers
def _num(value, digits=None):
    """float or None (never NaN/inf, so the result is always valid JSON)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return round(v, digits) if digits is not None else v


def _text(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def _percentile(series, value):
    """Percent of cells at or below this value (ties count half)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty or value is None:
        return None
    pct = ((s < value).sum() + 0.5 * (s == value).sum()) / len(s) * 100
    # clamp so wording like "hotter than 100% of cells" never appears
    return max(1, min(99, int(round(pct))))


def _level(pct):
    if pct is None:
        return None
    if pct >= HIGH_PCT:
        return "high"
    if pct <= LOW_PCT:
        return "low"
    return "middle"


def _reading(key, pct):
    if key == "lst":
        return f"Hotter than {pct}% of grid cells"
    if key == "ndvi":
        return f"Greener than {pct}% of grid cells"
    return f"More built-up or bare-surface signal than {pct}% of grid cells"


def _neighbour_stats(row, data):
    lat = _num(row.get("centroid_lat"))
    lon = _num(row.get("centroid_lon"))
    if lat is None or lon is None:
        return None
    lats = np.radians(pd.to_numeric(data["centroid_lat"], errors="coerce").to_numpy(dtype=float))
    lons = np.radians(pd.to_numeric(data["centroid_lon"], errors="coerce").to_numpy(dtype=float))
    lat_r, lon_r = math.radians(lat), math.radians(lon)
    a = np.sin((lats - lat_r) / 2) ** 2 + np.cos(lat_r) * np.cos(lats) * np.sin((lons - lon_r) / 2) ** 2
    dist = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    mask = (dist <= NEIGHBOUR_RADIUS_M) & (dist > 1)  # > 1 m excludes the cell itself
    lst = pd.to_numeric(data["lst_celsius"], errors="coerce").to_numpy(dtype=float)
    vals = lst[mask]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return {"n": 0, "mean": None}
    return {"n": int(vals.size), "mean": float(vals.mean())}


# ----------------------------------------------------------- why factors
# For each indicator: which side (low/high) is the "heating" side.
HEATING_SIDE = {"ndvi": "low", "ndbi": "high"}
OPPOSITE = {"low": "high", "high": "low"}


def _strength(lst_level, ind_level, heating_side):
    if lst_level == "high":
        if ind_level == heating_side:
            return "supports"
        if ind_level == OPPOSITE[heating_side]:
            return "does_not_support"
        return "mixed"
    if lst_level == "low":
        if ind_level == OPPOSITE[heating_side]:
            return "supports"
        if ind_level == heating_side:
            return "does_not_support"
        return "mixed"
    return "mixed"


def _vegetation_text(lst_level, strength, p):
    if lst_level == "middle":
        return (f"The surface temperature is mid-range, so vegetation (greener than {p}% "
                f"of cells) is not a clear driver here.")
    if lst_level == "high":
        if strength == "supports":
            return (f"The vegetation signal is lower than in most of the study area (greener than "
                    f"only {p}% of cells). This is consistent with less cooling from plants.")
        if strength == "does_not_support":
            return (f"The vegetation signal is higher than in most of the study area (greener than "
                    f"{p}% of cells), so low vegetation does not appear to explain the heat here.")
        return (f"The vegetation signal is mid-range (greener than {p}% of cells), so it is at "
                f"most a weak part of the explanation.")
    # low surface temperature
    if strength == "supports":
        return (f"The vegetation signal is higher than in most of the study area (greener than "
                f"{p}% of cells). This is consistent with more cooling from plants.")
    if strength == "does_not_support":
        return (f"The vegetation signal is lower than in most of the study area (greener than "
                f"only {p}% of cells), so vegetation does not appear to explain the cooler reading.")
    return (f"The vegetation signal is mid-range (greener than {p}% of cells), so it is at "
            f"most a weak part of the explanation.")


def _builtup_text(lst_level, strength, p):
    if lst_level == "middle":
        return (f"The surface temperature is mid-range, so the built-up signal (higher than {p}% "
                f"of cells) is not a clear driver here.")
    if lst_level == "high":
        if strength == "supports":
            return (f"The built-up signal is higher than in most of the study area (higher than "
                    f"{p}% of cells). This is consistent with more heat-absorbing built-up or "
                    f"bare, dry surfaces. NDBI cannot tell these two apart.")
        if strength == "does_not_support":
            return (f"The built-up signal is lower than in most of the study area (higher than "
                    f"only {p}% of cells), so built-up surfaces do not appear to explain the heat here.")
        return (f"The built-up signal is mid-range (higher than {p}% of cells), so it is at "
                f"most a weak part of the explanation.")
    if strength == "supports":
        return (f"The built-up signal is lower than in most of the study area (higher than only "
                f"{p}% of cells). This is consistent with fewer heat-absorbing surfaces.")
    if strength == "does_not_support":
        return (f"The built-up signal is higher than in most of the study area (higher than "
                f"{p}% of cells), so built-up surfaces do not appear to explain the cooler reading.")
    return (f"The built-up signal is mid-range (higher than {p}% of cells), so it is at "
            f"most a weak part of the explanation.")


def _neighbourhood_factor(lst, lst_level, stats):
    if lst is None or stats is None or stats["n"] < MIN_NEIGHBOURS or stats["mean"] is None:
        return {
            "factor": "surroundings", "label": "Surroundings", "strength": "info",
            "kind": KIND_POSSIBLE, "pattern": "unknown",
            "finding": "There are not enough nearby cells with valid data to compare this "
                       "cell with its surroundings.",
        }
    diff = lst - stats["mean"]
    n = stats["n"]
    if diff >= NEIGHBOUR_DIFF_C:
        pattern = "hot_spot"
        text = (f"This cell is {diff:.1f} °C hotter than the average of the {n} nearby cells "
                f"(within {NEIGHBOUR_RADIUS_M} m). A local difference like this points to features "
                f"specific to this spot rather than a wider warm area. This dataset cannot say which.")
    elif diff <= -NEIGHBOUR_DIFF_C:
        pattern = "cooler_pocket"
        text = (f"This cell is {abs(diff):.1f} °C cooler than the average of the {n} nearby cells "
                f"(within {NEIGHBOUR_RADIUS_M} m). A local difference like this points to features "
                f"specific to this spot. This dataset cannot say which.")
    else:
        pattern = "similar"
        area = {"high": "warm", "low": "cool"}.get(lst_level, "typical")
        text = (f"This cell is within {NEIGHBOUR_DIFF_C:g} °C of the {n} nearby cells, so it looks "
                f"like part of a broader {area} area rather than an isolated spot.")
    return {
        "factor": "surroundings", "label": "Surroundings", "strength": "info",
        "kind": KIND_POSSIBLE, "pattern": pattern, "difference_c": round(diff, 1),
        "neighbour_cells": n, "finding": text,
    }


def _weather_factor(row, data):
    """Only claim weather is uniform if the data really shows that."""
    cols = [c for c in WEATHER_COLUMNS if c in data.columns]
    if not cols:
        return None
    if data[cols].nunique(dropna=True).max() > 1:
        return None  # weather varies by cell: do not speculate
    return {
        "factor": "weather", "label": "Weather", "strength": "info", "kind": KIND_CONTEXT,
        "finding": "Weather values in this dataset are the same for every grid cell on the "
                   "observation date, so they cannot explain why this area differs from others.",
    }


# -------------------------------------------------------------- main API
def explain_grid(row, data, ml_prediction=None):
    """Build the explanation for one grid cell (a row of the dataset)."""
    values = {k: _num(row.get(meta["column"])) for k, meta in INDICATORS.items()}

    evidence, pcts, levels = [], {}, {}
    for key, meta in INDICATORS.items():
        col, v = meta["column"], values[key]
        if v is None or col not in data.columns:
            evidence.append({
                "key": key, "label": meta["label"], "value": None, "unit": meta["unit"],
                "city_median": None, "percentile": None, "level": None,
                "kind": KIND_UNAVAILABLE, "reading": "No valid measurement for this cell.",
            })
            pcts[key], levels[key] = None, None
            continue
        pct = _percentile(data[col], v)
        pcts[key], levels[key] = pct, _level(pct)
        evidence.append({
            "key": key, "label": meta["label"], "value": round(v, meta["digits"]),
            "unit": meta["unit"],
            "city_median": _num(pd.to_numeric(data[col], errors="coerce").median(), meta["digits"]),
            "percentile": pct, "level": levels[key], "kind": KIND_MEASURED,
            "reading": _reading(key, pct),
        })

    lst_level = levels["lst"]
    observed = _text(row.get("heat_risk"))
    ml = _text(ml_prediction)
    models_agree = None if (ml is None or observed is None) else (ml.lower() == observed.lower())

    # --- why this area may be hot / cool
    why = []
    if lst_level is not None:
        for key, label, text_fn in (
            ("ndvi", "Vegetation", _vegetation_text),
            ("ndbi", "Built-up surfaces", _builtup_text),
        ):
            if levels[key] is None:
                continue
            strength = _strength(lst_level, levels[key], HEATING_SIDE[key])
            why.append({
                "factor": key, "label": label, "strength": strength, "kind": KIND_POSSIBLE,
                "finding": text_fn(lst_level, strength, pcts[key]),
            })
    why.append(_neighbourhood_factor(values["lst"], lst_level, _neighbour_stats(row, data)))
    weather = _weather_factor(row, data)
    if weather:
        why.append(weather)

    # --- overview
    if values["lst"] is None or pcts["lst"] is None:
        headline = "No valid land surface temperature is available for this cell, so no explanation can be given."
    else:
        shown = observed if observed else "Unclassified"
        headline = (f"{shown} heat risk in this satellite snapshot: hotter than {pcts['lst']}% "
                    f"of grid cells in the study area.")
    if models_agree is True:
        model_note = "The ML model's prediction matches the observed class."
    elif models_agree is False:
        model_note = (f"The ML model predicts {ml} while the observed class is {observed}. "
                      f"Treat the explanation with extra care.")
    else:
        model_note = "No ML prediction is available for comparison."

    overview = {
        "headline": headline,
        "observed_risk": observed, "ml_prediction": ml, "models_agree": models_agree,
        "model_note": model_note,
        "heat_risk_score": _num(row.get("heat_risk_score"), 1),
        "risk_scale_note": ("Heat-risk classes describe how this area's surface temperature compares "
                            "within this satellite snapshot. They are not health or safety thresholds."),
    }

    # --- confidence in the explanation
    supports = sum(1 for w in why if w["factor"] in ("ndvi", "ndbi") and w["strength"] == "supports")
    against = sum(1 for w in why if w["factor"] in ("ndvi", "ndbi") and w["strength"] == "does_not_support")
    reasons = []
    if lst_level in ("high", "low") and supports >= 1 and against == 0 and models_agree is not False:
        level = "moderate"
        reasons.append("The vegetation and built-up signals point in the same direction as the surface temperature.")
    else:
        level = "weak"
        if lst_level is None:
            reasons.append("Surface temperature is missing for this cell.")
        elif lst_level == "middle":
            reasons.append("The surface temperature is mid-range, so no single factor stands out.")
        if against:
            reasons.append("At least one signal does not match the surface temperature, so other factors are likely involved.")
        if lst_level in ("high", "low") and supports == 0 and not against:
            reasons.append("The vegetation and built-up signals give only weak support.")
        if models_agree is False:
            reasons.append("The ML prediction disagrees with the observed class.")
    date = _text(row.get("date"))
    reasons.append(f"Based on one satellite observation date{f' ({date})' if date else ''}.")
    reasons.append("Only NDVI, NDBI and surface temperature are available; land-use and canopy data are not.")

    cls_counts = {}
    if "heat_risk" in data.columns:
        cls_counts = {str(k): int(v) for k, v in data["heat_risk"].value_counts().items()}

    return {
        "grid_id": _text(row.get("grid_id")),
        "location": {"lat": _num(row.get("centroid_lat"), 5), "lon": _num(row.get("centroid_lon"), 5)},
        "overview": overview,
        "evidence": evidence,
        "why": why,
        "explanation_confidence": {"level": level, "reasons": reasons},
        "data_notes": {
            "observation_date": date,
            "data_type": "Historical satellite observation, not live conditions.",
            "grid_cells": int(len(data)),
            "class_counts": cls_counts,
        },
        "not_available": list(NOT_AVAILABLE),
        "limitations": list(LIMITATIONS),
    }