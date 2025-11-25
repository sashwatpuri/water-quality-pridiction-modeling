# ============================
# Water Quality XGBoost Model
# ============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt

# 1️⃣ Load the cleaned dataset
data = pd.read_csv("water_potability_cleaned.csv")

# 2️⃣ Split features and target
X = data.drop("Potability", axis=1)
y = data["Potability"]

# 3️⃣ Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4️⃣ Initialize and train XGBoost model
model = XGBClassifier(
    n_estimators=300,        # number of trees
    max_depth=6,             # controls tree complexity
    learning_rate=0.05,      # smaller = slower but better accuracy
    subsample=0.9,           # random rows to prevent overfitting
    colsample_bytree=0.9,    # random features per tree
    gamma=0.2,               # regularization term
    reg_lambda=1,            # L2 regularization
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"    # prevents deprecation warnings
)

print("🚀 Training XGBoost model...")
model.fit(X_train, y_train)

# 5️⃣ Predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 6️⃣ Evaluation
print("\n✅ Model Evaluation:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7️⃣ Save model
joblib.dump(model, "Water_XGBoost_Model.pkl")
print("\n💾 Model saved successfully as Water_XGBoost_Model.pkl")

# 8️⃣ Feature importance visualization
plt.figure(figsize=(8,5))
plt.barh(X.columns, model.feature_importances_, color='skyblue')
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importance in Water Potability")
plt.tight_layout()
plt.show()
