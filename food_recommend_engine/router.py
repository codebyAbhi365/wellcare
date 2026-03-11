from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import joblib
import os
import numpy as np
import pandas as pd
from firebase_admin import firestore
from pathlib import Path

# Import the existing Firebase client if possible, or initialize local
# For simplicity in this engine-specific router, we'll assume it will be integrated with the main app's DB
# However, to keep it modular, let's try to import from the main ml_pipeline if it's already initialized
try:
    from agent.ml_pipeline import new_db
except ImportError:
    new_db = None

recommend_router = APIRouter(prefix="/recommend", tags=["Recommendation"])

# Load model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "body_impact_rf_v2.pkl"
try:
    model = joblib.load(MODEL_PATH)
except:
    model = None

# Reference Food Dataset
FOOD_DATABASE = [
    {"name": "Oatmeal with Berries", "category": "Low", "impact": 15},
    {"name": "Grilled Chicken Salad", "category": "Low", "impact": 20},
    {"name": "Steamed Vegetables with Tofu", "category": "Low", "impact": 10},
    {"name": "Apple & Handful of Almonds", "category": "Low", "impact": 12},
    {"name": "Quinoa & Roasted Veggies", "category": "Medium", "impact": 45},
    {"name": "Grilled Salmon & Asparagus", "category": "Medium", "impact": 40},
    {"name": "Whole Wheat Pasta", "category": "Medium", "impact": 50},
    {"name": "Double Cheese Pizza", "category": "High", "impact": 85},
    {"name": "Beef Burger & Fries", "category": "High", "impact": 90},
    {"name": "Deep Fried Chicken", "category": "High", "impact": 95},
]

class AssessmentRequest(BaseModel):
    user_id: str = "user123"
    date: str = "2026-03-09"
    assessment_name: str = "morning" # morning, afternoon, evening

@recommend_router.post("/assess")
async def assess_body_impact(req: AssessmentRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Recommendation model not loaded")
    if not new_db:
        raise HTTPException(status_code=500, detail="Firebase DB not initialized")

    doc_ref = new_db.collection("Body_Assessment").document(f"{req.user_id}_{req.date}")
    doc = doc_ref.get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail="Assessment document not found in Firebase")

    data = doc.to_dict()
    packets = data.get(req.assessment_name)

    if not packets or not isinstance(packets, list):
        raise HTTPException(status_code=404, detail=f"No data packets found for {req.assessment_name}")

    # Calculate Averages
    avg_data = {
        "ba_ratio": np.mean([p.get("ba_ratio", 0) for p in packets]),
        "heart_rate": np.mean([p.get("heart_rate", 0) for p in packets]),
        "hrv": np.mean([p.get("hrv", 0) for p in packets]),
        "temperature": np.mean([p.get("temperature", 0) for p in packets]),
        "sleep_hours": np.mean([p.get("sleep_hours", 0) for p in packets]),
        "hydration_level": np.mean([p.get("hydration_level", 0) for p in packets]),
        "heat_factor": np.mean([p.get("heat_factor", 0) for p in packets]),
    }

    # Predict Holistic Impact
    input_df = pd.DataFrame([avg_data])
    predicted_score = float(model.predict(input_df)[0])

    # Recommendation Logic
    if predicted_score > 60:
        recommendations = [f for f in FOOD_DATABASE if f["category"] == "Low"]
        status = "CRITICAL: High body impact. Recommend very light meals."
    elif predicted_score > 30:
        recommendations = [f for f in FOOD_DATABASE if f["category"] in ["Low", "Medium"]]
        status = "MODERATE: Body is under load. Recommend balanced meals."
    else:
        recommendations = FOOD_DATABASE
        status = "STABLE: Body state is excellent. Most meals are fine."

    # Prepare Report
    report = {
        "timestamp": firestore.SERVER_TIMESTAMP,
        "averages": {k: round(v, 4) for k, v in avg_data.items()},
        "holistic_impact_score": round(predicted_score, 2),
        "status": status,
        "recommended_foods": recommendations[:5]
    }

    # Save Report back to Firebase
    doc_ref.set({
        f"report_{req.assessment_name}": report
    }, merge=True)

    return {
        "status": "success",
        "predicted_score": round(predicted_score, 2),
        "averages": report["averages"],
        "recommendations": report["recommended_foods"]
    }
