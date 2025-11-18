# ==========================================
# 🚰 Optimized Water Potability XGBoost Model (Final Fixed)
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
import joblib
import matplotlib.pyplot as plt

# Quick version check (optional but recommended)
import xgboost
print(f"XGBoost version: {xgboost.__version__}")

# 1️⃣ Load dataset
data = pd.read_csv("water_potability_cleaned.csv")

# 2️⃣ Split features and target
X = data.drop("Potability", axis=1)
y = data["Potability"]

# 3️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 4️⃣ Initialize optimized XGBoost model
model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.85,
    colsample_bytree=0.85,
    gamma=0.1,
    reg_lambda=1.5,
    reg_alpha=0.5,
    scale_pos_weight=1.2,
    random_state=42,
    use_label_encoder=False,  # Note: This is deprecated in XGBoost 1.3+; consider removing if on newer version
    eval_metric="logloss"
)

# 5️⃣ Train with early stopping using the modern callback system
print("🚀 Training with early stopping (XGBoost callback fix)...")
model.fit(
    X_train, y_train,
    #eval_set=[(X_test, y_test)],
    #callbacks=[EarlyStopping(rounds=40, save_best=True)],  # ✅ Use callback instead of deprecated early_stopping_rounds
    #verbose=True  # Set to False if you want less output
)

# 6️⃣ Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 7️⃣ Evaluation
print("\n✅ Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 8️⃣ Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
print(f"\n📊 5-Fold CV Accuracy: {np.mean(cv_scores) * 100:.2f}% ± {np.std(cv_scores):.2f}")

# 9️⃣ Save model
joblib.dump(model, "Optimized_Water_XGBoost_Model.pkl")
print("\n💾 Model saved successfully as Optimized_Water_XGBoost_Model.pkl")

# 🔟 Feature importance visualization
plt.figure(figsize=(8, 5))
sorted_idx = np.argsort(model.feature_importances_)
plt.barh(X.columns[sorted_idx], model.feature_importances_[sorted_idx], color='teal')
plt.xlabel("Feature Importance")
plt.title("Optimized XGBoost Feature Importance")
plt.tight_layout()
plt.show()
