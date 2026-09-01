"""
Minimal FastAPI entry point for UrbanTherm.

This file only starts the API. Data collection, ML, and map features
will be added in later steps.
"""

from fastapi import FastAPI

app = FastAPI(
    title="UrbanTherm API",
    description="Hyperlocal urban heat intelligence and risk forecasting",
    version="0.1.0",
)


@app.get("/")
def read_root() -> dict[str, str]:
    """Health-style check so we can confirm the API is running."""
    return {"message": "UrbanTherm API is running"}
