import numpy as np
from firebase_admin import firestore
from firebase.reader import db, get_all_readings

# ─────────────────────────────────────────────
# 1. CORE DERIVATION (Matches your image values)
# ─────────────────────────────────────────────

def calculate_spike_from_derivation(d: dict, baseline: dict) -> float:
    """Calculates spike index (0-100) using relative baseline changes."""
    hr   = float(d.get("heart_rate") or 70)
    hrv  = float(d.get("hrv") or 60)
    amp  = float(d.get("pulse_amplitude") or 1.0)
    temp = float(d.get("skin_temperature") or 36.0)
    bvp  = float(d.get("blood_volume_pulse_intensity") or 1.0)

    # Normalized changes relative to baseline
    hr_change   = (hr - baseline["b_hr"]) / (baseline["b_hr"] + 1e-6)
    hrv_drop    = (baseline["b_hrv"] - hrv) / (baseline["b_hrv"] + 1e-6)
    amp_change  = (amp - baseline["b_amp"]) / (baseline["b_amp"] + 1e-6)
    temp_change = (temp - baseline["b_temp"]) / (baseline["b_temp"] + 1e-6)
    bvp_change  = (bvp - baseline["b_bvp"]) / (baseline["b_bvp"] + 1e-6)

    # Weighted model based on your research path
    spike_score = (
        0.30 * hrv_drop    +
        0.25 * amp_change  +
        0.20 * hr_change   +
        0.15 * bvp_change  +
        0.10 * temp_change
    ) * 100

    return round(max(0, min(spike_score, 100)), 2)

# ─────────────────────────────────────────────
# 2. DASHBOARD METRICS (As seen in image footer)
# ─────────────────────────────────────────────

def compute_dashboard_metrics(readings: list[dict], baseline: dict, spike_data: dict) -> dict:
    """Calculates max peaks/drops and duration counts."""
    max_hrv_drop = 0.0
    max_amp_change = 0.0
    max_bvp_intensity = 0.0
    max_hr_peak = 0.0
    temp_rise_count = 0
    
    spike_values = spike_data["spike_values"]

    for i, d in enumerate(readings):
        # Current Values
        hr   = float(d.get("heart_rate") or baseline["b_hr"])
        hrv  = float(d.get("hrv") or baseline["b_hrv"])
        amp  = float(d.get("pulse_amplitude") or baseline["b_amp"])
        bvp  = float(d.get("blood_volume_pulse_intensity") or baseline["b_bvp"])
        temp = float(d.get("skin_temperature") or baseline["b_temp"])

        # Metric Calculations (Relative %)
        hrv_drop_pct = ((baseline["b_hrv"] - hrv) / baseline["b_hrv"]) * 100
        amp_chg_pct  = (abs(amp - baseline["b_amp"]) / baseline["b_amp"]) * 100
        bvp_int_pct  = (abs(bvp - baseline["b_bvp"]) / baseline["b_bvp"]) * 100
        hr_peak_pct  = ((hr - baseline["b_hr"]) / baseline["b_hr"]) * 100

        if hrv_drop_pct  > max_hrv_drop:   max_hrv_drop = hrv_drop_pct
        if amp_chg_pct   > max_amp_change: max_amp_change = amp_chg_pct
        if bvp_int_pct   > max_bvp_intensity: max_bvp_intensity = bvp_int_pct
        if hr_peak_pct   > max_hr_peak:    max_hr_peak = hr_peak_pct
        
        # Inflammation Watch (Threshold based on temp rise)
        if (temp - baseline["b_temp"]) > 0.3:
            temp_rise_count += 1

    return {
        "hrv_drop_pct":            round(max_hrv_drop, 1),
        "pulse_amp_change_pct":    round(max_amp_change, 1),
        "bvp_intensity_pct":       round(max_bvp_intensity, 1),
        "hr_peak_pct":             round(max_hr_peak, 1),
        "inflammation_watch_mins": temp_rise_count,
        "max_spike_index":         spike_data["max_spike"],
        "avg_spike_index":         spike_data["avg_spike"]
    }

# ─────────────────────────────────────────────
# 3. FIREBASE WRITERS (As per your data paths)
# ─────────────────────────────────────────────

def write_spike_array_to_firebase(user_id: str, spike_data: dict):
    """Stores the full 120-value array for Chart.js."""
    doc_ref = db.collection("glucose").document(user_id).collection("spike_graph").document("history")
    doc_ref.set({
        "timestamps":   spike_data["timestamps"],
        "spike_values": spike_data["spike_values"], # Array for graph
        "max_spike":    spike_data["max_spike"],
        "computed_at":  firestore.SERVER_TIMESTAMP
    })

def write_summary_to_firebase(user_id: str, metrics: dict, baseline: dict):
    """Stores dashboard footer metrics."""
    doc_ref = db.collection("glucose").document(user_id).collection("summary").document("latest")
    doc_ref.set({
        **metrics, # Spreads hrv_drop_pct, bvp_intensity_pct, etc.
        "baseline_hr":  round(baseline["b_hr"], 2),
        "baseline_hrv": round(baseline["b_hrv"], 2),
        "computed_at":  firestore.SERVER_TIMESTAMP
    })

# ─────────────────────────────────────────────
# 4. FULL PIPELINE
# ─────────────────────────────────────────────

def process_and_push(user_id: str) -> dict:
    readings = get_all_readings(user_id)
    if not readings: return {"error": "No data found"}

    # Sort data to ensure first 10 are the start
    readings = sorted(readings, key=lambda x: x.get("_doc_id", ""))
    
    # Calculate Baseline (First 10 readings)
    base_slice = readings[:10]
    baseline = {
        "b_hr":  float(np.mean([r.get("heart_rate") or 70 for r in base_slice])),
        "b_hrv": float(np.mean([r.get("hrv") or 60 for r in base_slice])),
        "b_amp": float(np.mean([r.get("pulse_amplitude") or 1.0 for r in base_slice])),
        "b_bvp": float(np.mean([r.get("blood_volume_pulse_intensity") or 1.0 for r in base_slice])),
        "b_temp":float(np.mean([r.get("skin_temperature") or 36.0 for r in base_slice])),
    }

    # Step 1: Generate Graph Array (120 values)
    spike_values = [calculate_spike_from_derivation(r, baseline) for r in readings]
    timestamps = [r.get("_doc_id") or "unknown" for r in readings]
    
    spike_data = {
        "timestamps":   timestamps,
        "spike_values": spike_values,
        "max_spike":    max(spike_values) if spike_values else 0,
        "avg_spike":    round(float(np.mean(spike_values)), 2) if spike_values else 0
    }

    # Step 2: Compute Dashboard Metrics
    metrics = compute_dashboard_metrics(readings, baseline, spike_data)

    # Step 3: Write to Firebase
    write_spike_array_to_firebase(user_id, spike_data)
    write_summary_to_firebase(user_id, metrics, baseline)

    return {
        "status": "success",
        "user_id": user_id,
        "data_preview": metrics
    }


def write_spike_array_to_firebase(user_id: str, spike_data: dict):
    """
    Explicitly stores parallel arrays for Chart.js.
    Path: glucose/{user_id}/spike_graph/history
    """
    doc_ref = (
        db.collection("glucose")
          .document(user_id)
          .collection("spike_graph")
          .document("history")
    )
    
    # This structure maps directly to your Chart.js requirements
    doc_ref.set({
        "timestamps":   spike_data["timestamps"],   # X-Axis Array
        "spike_values": spike_data["spike_values"], # Y-Axis Array
        "max_spike":    spike_data["max_spike"],
        "avg_spike":    spike_data["avg_spike"],
        "computed_at":  firestore.SERVER_TIMESTAMP,
    })