import joblib
import pandas as pd
import os
import numpy as np

# Change directory to the script's folder to ensure paths work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1. Load the trained model
MODEL_PATH = "body_impact_rf_v2.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# 2. Reference Food Dataset
# Each food has a base metabolic cost category
FOOD_DATABASE = [
    {"name": "Oatmeal with Berries", "category": "Low", "description": "Light and fiber-rich, easy on the metabolism."},
    {"name": "Grilled Chicken Salad", "category": "Low", "description": "High protein, low carb, very stable response."},
    {"name": "Steamed Vegetables with Tofu", "category": "Low", "description": "Nutrient dense, minimal glycemic load."},
    {"name": "Apple & Handful of Almonds", "category": "Low", "description": "Simple healthy snack, low insulin demand."},
    {"name": "Quinoa & Roasted Veggies", "category": "Medium", "description": "Balanced carbs and fiber."},
    {"name": "Grilled Salmon & Asparagus", "category": "Medium", "description": "Healthy fats and protein."},
    {"name": "Whole Wheat Pasta", "category": "Medium", "description": "Steady energy release."},
    {"name": "Double Cheese Pizza", "category": "High", "description": "High fat/carb combo, high metabolic cost."},
    {"name": "Beef Burger & Fries", "category": "High", "description": "Heavy load, likely to cause significant spikes."},
    {"name": "Deep Fried Chicken", "category": "High", "description": "Inflammatory oils and high caloric density."},
]

def recommend_food(data_packets):
    """
    Takes 100 body data packets, averages them, and recommends food.
    """
    if not model:
        return {"error": "Model not loaded"}
    
    if len(data_packets) == 0:
        return {"error": "No data packets provided"}

    # Average the metrics from 100 packets
    avg_data = {
        "ba_ratio": np.mean([p.get("ba_ratio", 0) for p in data_packets]),
        "heart_rate": np.mean([p.get("heart_rate", 0) for p in data_packets]),
        "hrv": np.mean([p.get("hrv", 0) for p in data_packets]),
        "temperature": np.mean([p.get("temperature", 0) for p in data_packets]),
        "sleep_hours": np.mean([p.get("sleep_hours", 0) for p in data_packets]),
        "hydration_level": np.mean([p.get("hydration_level", 0) for p in data_packets]),
        "heat_factor": np.mean([p.get("heat_factor", 0) for p in data_packets]),
    }

    # Predict Holistic Impact Score
    input_df = pd.DataFrame([avg_data])
    predicted_score = model.predict(input_df)[0]
    
    # Recommendation Logic
    # Scale: 0-100 (based on training data range)
    # Low stress: < 30, Medium Stress: 30-60, High Stress: > 60
    
    recommendations = []
    if predicted_score > 60:
        # High Impact/Stress detected: Recommend only Low category
        recommendations = [f for f in FOOD_DATABASE if f["category"] == "Low"]
        status = "CRITICAL: High body impact detected. Recommend very light meals."
    elif predicted_score > 30:
        # Medium Impact: Recommend Low and Medium
        recommendations = [f for f in FOOD_DATABASE if f["category"] in ["Low", "Medium"]]
        status = "MODERATE: Body is under some load. Recommend balanced, easy-to-digest meals."
    else:
        # Low Impact: Recommend anything
        recommendations = FOOD_DATABASE
        status = "STABLE: Body state is excellent. Most meals are fine."

    return {
        "predicted_impact_score": round(predicted_score, 2),
        "body_status": status,
        "recommendations": recommendations[:5] # Return top 5
    }

# --- TEST ---
if __name__ == "__main__":
    # Mocking 100 packets of "High Impact" state
    mock_packets = [
        {
            "ba_ratio": 0.5, "heart_rate": 95, "hrv": 35, "temperature": 37.5,
            "sleep_hours": 5, "hydration_level": 45, "heat_factor": 8.5
        } for _ in range(100)
    ]
    
    result = recommend_food(mock_packets)
    print(f"Predicted Score: {result['predicted_impact_score']}")
    print(f"Status: {result['body_status']}")
    print("Top Recommendations:")
    for r in result['recommendations']:
        print(f" - {r['name']} ({r['category']})")
