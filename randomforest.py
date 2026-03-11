import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
df = pd.read_csv("food_impact_data_nofood.csv")

# Separate features and target
X = df.drop("impact", axis=1)
y = df["impact"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))


importances = model.feature_importances_
feature_names = X.columns

# plt.barh(feature_names, importances)
# plt.xlabel("Importance")
# plt.title("Feature Importance")
# plt.show()


# joblib.dump(model, "rf_model.pkl")
# print("Model saved successfully.")

sample = X.iloc[0:1]
prediction = model.predict(sample)

print("Prediction:", prediction)