import random
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from agent.ml_pipeline import new_db
    from firebase_admin import firestore
except ImportError:
    print("Error: Could not import Firebase database from agent.ml_pipeline")
    exit(1)

def generate_packets(count=100, state="normal"):
    packets = []
    for _ in range(count):
        if state == "high_stress":
            packet = {
                "ba_ratio": round(random.uniform(0.45, 0.55), 2),
                "heart_rate": random.randint(90, 110),
                "hrv": random.randint(20, 40),
                "temperature": round(random.uniform(37.2, 38.0), 1),
                "sleep_hours": round(random.uniform(4, 6), 1),
                "hydration_level": random.randint(40, 60),
                "heat_factor": round(random.uniform(7, 10), 2)
            }
        else:
            packet = {
                "ba_ratio": round(random.uniform(0.3, 0.4), 2),
                "heart_rate": random.randint(60, 80),
                "hrv": random.randint(60, 90),
                "temperature": round(random.uniform(36.0, 36.8), 1),
                "sleep_hours": round(random.uniform(7, 9), 1),
                "hydration_level": random.randint(80, 100),
                "heat_factor": round(random.uniform(0, 3), 2)
            }
        packets.append(packet)
    return packets

def seed_data(user_id="user123", date="2026-03-09"):
    if not new_db:
        print("Error: Firebase DB not initialized")
        return

    print(f"Seeding mock assessment data for {user_id} on {date}...")
    
    doc_ref = new_db.collection("Body_Assessment").document(f"{user_id}_{date}")
    
    data = {
        "morning": generate_packets(state="normal"),
        "afternoon": generate_packets(state="normal"),
        "evening": generate_packets(state="high_stress"), # Mocking some evening load
        "last_updated": firestore.SERVER_TIMESTAMP
    }
    
    doc_ref.set(data)
    print("SUCCESS: Mock data seeded to Body_Assessment collection.")

if __name__ == "__main__":
    seed_data()
