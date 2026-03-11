import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Load data
DATA_PATH = "body_impact_training_data.csv"
if not os.path.exists(DATA_PATH):
    print(f"Error: {DATA_PATH} not found.")
    exit(1)

df = pd.read_csv(DATA_PATH)

# Use individual values and skip empty rows if any
df = df.dropna()

# 2. Features and Target
# ba_ratio,heart_rate,hrv,temperature,sleep_hours,hydration_level,heat_factor,impact_score
X = df[['ba_ratio', 'heart_rate', 'hrv', 'temperature', 'sleep_hours', 'hydration_level', 'heat_factor']]
y = df['impact_score']

# 3. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
print("Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# 6. Save Model
MODEL_NAME = "body_impact_rf_v2.pkl"
joblib.dump(model, MODEL_NAME)
print(f"Model saved as {MODEL_NAME}")
