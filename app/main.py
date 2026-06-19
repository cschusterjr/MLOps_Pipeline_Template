import pathlib
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_PATH = pathlib.Path("artifacts/model.pkl")
FEATURE_COLUMNS_PATH = pathlib.Path("artifacts/feature_columns.pkl")
METRICS_PATH = pathlib.Path("artifacts/metrics.json")


app = FastAPI(
    title="Student Success Prediction API",
    description="An MLOps-style API that predicts a student's final course grade using academic performance signals.",
    version="2.0.0",
)


model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)


class StudentFeatures(BaseModel):
    attendance_rate: float = Field(..., ge=0, le=100)
    quiz_average: float = Field(..., ge=0, le=100)
    homework_average: float = Field(..., ge=0, le=100)
    exam_average: float = Field(..., ge=0, le=100)
    missing_assignments: int = Field(..., ge=0)
    late_submissions: int = Field(..., ge=0)
    engagement_score: float = Field(..., ge=0, le=100)
    current_grade: float = Field(..., ge=0, le=100)


class PredictionResponse(BaseModel):
    predicted_final_grade: float
    risk_level: str
    recommendation: str


def classify_risk(predicted_grade: float) -> str:
    if predicted_grade < 60:
        return "High"
    if predicted_grade < 70:
        return "Medium"
    return "Low"


def recommend_action(risk_level: str) -> str:
    if risk_level == "High":
        return "Immediate intervention recommended. Review missing assignments, exam performance, and attendance patterns."
    if risk_level == "Medium":
        return "Monitor closely and provide targeted academic support before the student falls further behind."
    return "Student appears on track. Continue monitoring performance and engagement."


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    if METRICS_PATH.exists():
        metrics = pd.read_json(METRICS_PATH, typ="series").to_dict()
    else:
        metrics = {}

    return {
        "model_type": type(model).__name__,
        "features": feature_columns,
        "metrics": metrics,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: StudentFeatures):
    input_data = pd.DataFrame([payload.model_dump()])[feature_columns]

    prediction = float(model.predict(input_data)[0])
    prediction = round(max(0, min(100, prediction)), 1)

    risk_level = classify_risk(prediction)

    return PredictionResponse(
        predicted_final_grade=prediction,
        risk_level=risk_level,
        recommendation=recommend_action(risk_level),
    )