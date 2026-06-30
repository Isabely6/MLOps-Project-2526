"""Pydantic request/response schemas for the serving API.

Request fields use friendly lowercase ``snake_case`` names (matching the project
guidelines' example input). They are mapped to the model's training column names in
``main.py`` via :data:`FIELD_TO_MODEL_COLUMN`, so the API contract stays clean while the
underlying sklearn Pipeline still receives the exact columns it was trained on.
"""

from typing import Dict

from pydantic import BaseModel, ConfigDict, Field


# Maps API request field -> the column name the serving model was trained on.
# The order here is the canonical model feature order.
FIELD_TO_MODEL_COLUMN = {
    "population": "Population",
    "gdp_per_capita": "GDP_per_capita",
    "life_expectancy": "Life_expectancy",
    "health_expenditure": "Health_expenditure",
    "internet_usage": "Internet_usage",
    "co2_emissions": "CO2_emissions",
}


class PredictRequest(BaseModel):
    """One country's pre-pandemic socioeconomic / health indicators."""

    population: float = Field(..., gt=0, description="Total population (persons).")
    gdp_per_capita: float = Field(..., ge=0, description="GDP per capita (current US$).")
    life_expectancy: float = Field(
        ..., ge=0, le=120, description="Average life expectancy at birth (years)."
    )
    health_expenditure: float = Field(
        ..., ge=0, le=100, description="Total health expenditure (% of GDP)."
    )
    internet_usage: float = Field(
        ..., ge=0, le=100, description="Individuals using the internet (per 100 people)."
    )
    co2_emissions: float = Field(
        ..., ge=0, description="CO2 emissions per capita (tons)."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "population": 46000000,
                "gdp_per_capita": 32000,
                "life_expectancy": 81.0,
                "health_expenditure": 9.5,
                "internet_usage": 85.0,
                "co2_emissions": 5.8,
            }
        }
    )


class PredictResponse(BaseModel):
    mortality_severity: str = Field(..., description="Predicted severity tier.")
    probabilities: Dict[str, float] = Field(
        ..., description="Per-class probabilities, ordered Low/Medium/High."
    )
    model_version: str
    latency_ms: float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "mortality_severity": "High",
                "probabilities": {"Low": 0.31, "Medium": 0.21, "High": 0.48},
                "model_version": "1.0.0",
                "latency_ms": 2.4,
            }
        }
    )


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    ready: bool
    detail: str


class ModelInfoResponse(BaseModel):
    model_version: str
    model_type: str
    feature_names: list
    label_order: list
    metrics: dict
    trained_at: str
