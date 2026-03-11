import sys
import os
from pathlib import Path

# Add the project directory to sys.path
project_dir = Path("c:/Users/abhishek dipak kadam/OneDrive/Desktop/WellCare v2.o/skin_agent")
sys.path.append(str(project_dir))

try:
    from agent.ml_pipeline import extract_features
    import numpy as np

    # Mock data
    readings = [
        {"heart_rate": 80, "hrv": 50, "b_a_ratio": 1.1, "skin_temperature": 36.5, "_doc_id": "1"},
        {"heart_rate": 90, "hrv": 45, "b_a_ratio": 1.2, "skin_temperature": 37.0, "_doc_id": "2"},
        {"heart_rate": 100, "hrv": 40, "b_a_ratio": 1.3, "skin_temperature": 37.5, "_doc_id": "3"},
    ]
    baseline = {"baseline_hr": 70, "baseline_hrv": 60, "baseline_ba": 1.0, "baseline_temp": 36.0}

    features = extract_features(readings, baseline, "2026-03-08T14:40:46")
    
    print(f"Calculated Features: {features}")
    
    if "heat_factor" in features:
        print(f"SUCCESS: Heat Factor found: {features['heat_factor']}")
        # Manual verification of the values based on mock data
        # max_hr_spike = 100 - 70 = 30
        # hr_strain = 30 / (180 - 70) = 30 / 110 = 0.2727
        # ba_diffs = [0.1, 0.2, 0.3], avg_ba_shift = 0.2
        # ba_strain = 0.2 / 1.5 = 0.1333
        # max_temp_shift = 37.5 - 36.0 = 1.5
        # heat_factor = (5 * 0.2727) + (5 * 0.1333) + (1.5 * 2) = 1.3635 + 0.6665 + 3.0 = 5.03
        print(f"Expected Heat Factor ~5.03")
    else:
        print("FAILURE: Heat Factor NOT found in features.")

except Exception as e:
    print(f"Error during verification: {e}")
    import traceback
    traceback.print_exc()
