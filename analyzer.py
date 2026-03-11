# agent/analyzer.py

THRESHOLDS = {
    # metric                     (warn,  alert)
    "hrv_drop_pct":             (40,    65),
    "pulse_amp_change_pct":     (50,    80),
    "bvp_intensity_pct":        (100,   150),
    "hr_peak_pct":              (20,    35),
    "inflammation_watch_mins":  (40,    60),
    "spike_duration_mins":      (40,    70),
    "latest_spike_index":       (95,    120),
    "max_consecutive_spike":    (15,    30),   # sustained spike streak

    # Sigma thresholds — >1.5 sigma = unusual for THIS user
    # >2.5 sigma = very abnormal for this user specifically
    "hrv_sigma":                (2.0,   3.0),
    "bvp_sigma":                (2.0,   3.0),
    "hr_sigma":                 (2.0,   3.0),
    "temp_sigma":               (2.0,   3.0),
}

SKIN_EFFECTS = {
    "hrv_drop_pct": (
        "HRV drop → cortisol surge → excess sebum → clogged pores → acne risk in 24–48 hrs"
    ),
    "pulse_amp_change_pct": (
        "Vascular stress → poor microcirculation → dull skin, slow healing, persistent blemishes"
    ),
    "bvp_intensity_pct": (
        "Glucose spike proxy → insulin + IGF-1 surge → sebaceous gland overstimulation → breakout risk"
    ),
    "hr_peak_pct": (
        "Metabolic overload → oxidative stress → skin redness, sensitivity, inflammation"
    ),
    "inflammation_watch_mins": (
        "Prolonged temp elevation → skin barrier damage → redness and acne flare risk"
    ),
    "spike_duration_mins": (
        "Extended metabolic stress window → sustained sebum overproduction → breakout risk"
    ),
    "latest_spike_index": (
        "Current spike elevated → active vascular + metabolic stress → skin inflammation right now"
    ),
    "max_consecutive_spike": (
        "Sustained spike streak → body hasn't recovered → compounding inflammation risk"
    ),
    "hrv_sigma": (
        "HRV is unusually low FOR YOU specifically → personal stress threshold exceeded → skin impact likely"
    ),
    "bvp_sigma": (
        "BVP unusually high FOR YOU → your vascular system under more stress than normal → inflammation risk"
    ),
    "hr_sigma": (
        "Heart rate unusually elevated FOR YOU → metabolic load beyond your personal normal"
    ),
    "temp_sigma": (
        "Skin temp unusually high FOR YOU → active inflammation beyond your personal baseline"
    ),
}

SEVERITY_ORDER = {"LOW": 0, "MODERATE": 1, "HIGH": 2}


def analyze_readings(metrics: dict) -> list[dict]:
    anomalies = []

    for metric, (warn, alert) in THRESHOLDS.items():
        value = metrics.get(metric)
        if value is None:
            continue

        if value >= alert:
            severity = "HIGH"
        elif value >= warn:
            severity = "MODERATE"
        else:
            continue

        anomalies.append({
            "metric":      metric,
            "value":       round(float(value), 2),
            "severity":    severity,
            "skin_effect": SKIN_EFFECTS.get(metric, "May affect skin health"),
        })

    anomalies.sort(key=lambda a: SEVERITY_ORDER.get(a["severity"], 0), reverse=True)
    return anomalies


def get_overall_risk(anomalies: list[dict]) -> str:
    if not anomalies:
        return "LOW"
    return max(anomalies, key=lambda a: SEVERITY_ORDER.get(a["severity"], 0))["severity"]