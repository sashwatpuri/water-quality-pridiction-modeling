from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pandas as pd

# Load cleaned dataset
data = pd.read_csv("water_potability.csv")
X = data.drop("Potability", axis=1)
y = data["Potability"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,   # number of trees
    max_depth=10,       # prevent overfitting
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "Water_RF_Model.pkl")
print("✅ Random Forest Model saved successfully!")

import matplotlib.pyplot as plt

importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
plt.barh(features, importance, color='teal')
plt.xlabel("Importance Score")
plt.title("Feature Importance in Water Potability Prediction")
plt.show()
