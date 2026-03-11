import firebase_admin
from firebase_admin import credentials, firestore
import os
import numpy as np
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

ml_router = APIRouter()

# 1. Initialize DB with the new credentials
CRED_PATH_NEW = os.path.join(os.path.dirname(__file__), "..", "wellcare-db-firebase.json")
try:
    new_app = firebase_admin.get_app('new_db')
except ValueError:
    if os.path.exists(CRED_PATH_NEW):
        cred = credentials.Certificate(CRED_PATH_NEW)
        new_app = firebase_admin.initialize_app(cred, name='new_db')
    else:
        new_app = None # Handle later if missing

if new_app:
    new_db = firestore.client(app=new_app)
else:
    new_db = None

# 2. Load RF Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "rf_model.pkl")
if os.path.exists(MODEL_PATH):
    rf_model = joblib.load(MODEL_PATH)
else:
    rf_model = None

class MealImpactRequest(BaseModel):
    user_id: str
    date: str
    meal_type: str
    sleep_score: int = 80
    hydration: int = 80

def extract_features(readings, baseline, meal_start_timestamp):
    # Sort readings by time
    readings = sorted(readings, key=lambda x: x.get("_doc_id", ""))
    
    if isinstance(baseline, list):
        baseline = baseline[0] if len(baseline) > 0 else {}
    elif baseline is None:
        baseline = {}
        
    # baseline values
    b_hr = float(baseline.get("baseline_hr", 70))
    b_hrv = float(baseline.get("baseline_hrv", 60))
    b_ba = float(baseline.get("baseline_ba", 1.0))
    b_temp = float(baseline.get("baseline_temp", 36.0))

    # lists for calculations
    hrs = [float(r.get("heart_rate") or b_hr) for r in readings]
    hrvs = [float(r.get("hrv") or b_hrv) for r in readings]
    bas = [float(r.get("b_a_ratio", r.get("dicrotic_notch_index", b_ba))) for r in readings]
    temps = [float(r.get("skin_temperature") or b_temp) for r in readings]
    
    # Check max value logic, handling empty lists safely
    if not readings:
        return {}
        
    sampling_interval = 1 # assume 1 minute between readings
    
    # HR Features
    hr_diffs = [hr - b_hr for hr in hrs]
    max_hr_spike = max(hr_diffs) if hr_diffs else 0
    time_to_peak_hr = hrs.index(max(hrs)) * sampling_interval if hrs else 0
    avg_hr_increase = np.mean(hr_diffs) if hr_diffs else 0
    area_under_hr_curve = sum(hr_diffs) * sampling_interval
    
    # HRV Features
    max_hrv_drop = b_hrv - min(hrvs) if hrvs else 0
    duration_hrv_below_baseline = sum(1 for hrv in hrvs if hrv < b_hrv) * sampling_interval
    
    # Recovery time HRV
    recovery_time_hrv = 0
    if hrvs:
        min_hrv_idx = hrvs.index(min(hrvs))
        consecutive_count = 0
        for i in range(min_hrv_idx, len(hrvs)):
            if hrvs[i] >= 0.95 * b_hrv:
                consecutive_count += 1
                if consecutive_count >= 5:
                    recovery_time_hrv = (i - min_hrv_idx) * sampling_interval
                    break
            else:
                consecutive_count = 0
                
    # B/A Features
    ba_diffs = [ba - b_ba for ba in bas]
    max_ba_shift = max(ba_diffs) if ba_diffs else 0
    avg_ba_shift = np.mean(ba_diffs) if ba_diffs else 0
    duration_ba_elevated = sum(1 for ba in bas if ba > b_ba + 0.02) * sampling_interval
    
    # Temperature Features
    temp_diffs = [temp - b_temp for temp in temps]
    max_temp_shift = max(temp_diffs) if temp_diffs else 0
    avg_temp_shift = np.mean(temp_diffs) if temp_diffs else 0
    
    # Heat Factor Calculation (Scales metabolic & thermal strain)
    # 1. Cardiovascular Component (Normalized Heart Rate Shift)
    hr_limit = 180
    hr_strain = (max_hr_spike) / (hr_limit - b_hr) if (hr_limit - b_hr) > 0 else 0
    
    # 2. Vascular Component (B/A Ratio Shift)
    ba_strain = abs(avg_ba_shift) / 1.5
    
    # 3. Thermogenic Component (Skin Temp)
    temp_strain = max_temp_shift / (39.5 - b_temp) if (39.5 - b_temp) > 0 else 0

    # 4. Final Heat Factor Calculation (Scale 0-10)
    # Weighting: HR (50%) + Vascular (50%) + Absolute Temp Spike
    heat_factor = (5 * hr_strain) + (5 * ba_strain) + (max_temp_shift * 2)
    heat_factor = max(0, min(10, heat_factor))
    
    return {
        "max_hr_spike": round(max_hr_spike, 2),
        "time_to_peak_hr": time_to_peak_hr,
        "avg_hr_increase": round(avg_hr_increase, 2),
        "area_under_hr_curve": round(area_under_hr_curve, 2),
        "max_hrv_drop": round(max_hrv_drop, 2),
        "duration_hrv_below_baseline": duration_hrv_below_baseline,
        "recovery_time_hrv": recovery_time_hrv,
        "max_ba_shift": round(max_ba_shift, 4),
        "avg_ba_shift": round(avg_ba_shift, 4),
        "duration_ba_elevated": duration_ba_elevated,
        "max_temp_shift": round(max_temp_shift, 2),
        "avg_temp_shift": round(avg_temp_shift, 2),
        "heat_factor": round(heat_factor, 2),
        "baseline_hrv": round(b_hrv, 2)
    }

@ml_router.post("/analyze_meal_impact")
async def analyze_meal_impact(req: MealImpactRequest):
    if not new_db:
        raise HTTPException(status_code=500, detail="New Firebase DB not initialized")
    if not rf_model:
        raise HTTPException(status_code=500, detail="RF Model not loaded")
        
    doc_path = f"Food_log/{req.user_id}_{req.date}"
    doc_ref = new_db.collection("Food_log").document(f"{req.user_id}_{req.date}")
    
    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail=f"Document not found at {doc_path}")
        
    data = doc.to_dict()
    
    # user spec: meal_type like "Breakfast", we expect "Breakfast" list & "baseline_Breakfast" dict
    meal_readings = data.get(req.meal_type)
    baseline_data = data.get(f"baseline_{req.meal_type}")
    
    if not meal_readings or not baseline_data:
        raise HTTPException(status_code=404, detail=f"Missing {req.meal_type} or baseline_{req.meal_type} data in document")
        
    # meal_start_timestamp might be from the first reading
    meal_start_timestamp = meal_readings[0].get("timestamp") if meal_readings else None
    
    # 1. Extract rigorous features
    features = extract_features(meal_readings, baseline_data, meal_start_timestamp)
    if not features:
        raise HTTPException(status_code=400, detail="Failed to extract features")
        
    # 2. Map required inputs for RF model
    # Model expects: hr_spike, hrv_drop, ba_shift, temp_shift, sleep_score, hydration, baseline_hrv
    model_input = pd.DataFrame([{
        "hr_spike": features["max_hr_spike"],
        "hrv_drop": features["max_hrv_drop"],
        "ba_shift": features["max_ba_shift"],
        "temp_shift": features["max_temp_shift"],
        "sleep_score": req.sleep_score,
        "hydration": req.hydration,
        "baseline_hrv": features["baseline_hrv"]
    }])
    
    # Predict impact
    impact_score = int(rf_model.predict(model_input)[0])
    
    # Save the features back to the database in the same document
    results_key = f"analyzed_{req.meal_type}"
    doc_ref.set({
        results_key: {
            **features,
            "sleep_score": req.sleep_score,
            "hydration": req.hydration,
            "impact_score": impact_score
        }
    }, merge=True)
    
    return {
        "status": "success",
        "doc_updated": doc_path,
        "results": {
            **features,
            "impact_score": impact_score
        }
    }
