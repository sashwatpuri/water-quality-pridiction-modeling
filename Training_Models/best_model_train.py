# ==========================================
# 🌲 Optimized Water Potability Random Forest Model (v2.0 - HPT + SMOTE)
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("🌲 Random Forest Water Potability Classifier with HPT\n")

# 1️⃣ Load dataset
data = pd.read_csv("water_potability_cleaned.csv")

# 2️⃣ Split features and target
X = data.drop("Potability", axis=1)
y = data["Potability"]

print(f"Dataset shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}\n")

# 3️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 4️⃣ Balance training set with SMOTE
print("⚖️ Applying SMOTE for class balancing...")
sm = SMOTE(random_state=42, k_neighbors=5)
X_res, y_res = sm.fit_resample(X_train, y_train)
print(f"   Original training set size: {X_train.shape[0]}")
print(f"   Resampled training set size: {X_res.shape[0]}")
print(f"   Class distribution after SMOTE: {np.bincount(y_res)}\n")

# 5️⃣ Hyperparameter search space for Random Forest
param_dist = {
    "n_estimators": [100, 200, 300, 500, 800],
    "max_depth": [10, 15, 20, 25, 30, None],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False],
    "class_weight": ["balanced", "balanced_subsample", None],
    "criterion": ["gini", "entropy"],
}

# 6️⃣ Base estimator
base_est = RandomForestClassifier(random_state=42, n_jobs=-1)

# 7️⃣ RandomizedSearchCV for hyperparameter tuning
print("🔎 Running RandomizedSearchCV (50 iterations with 5-Fold CV)...")
print("   This may take 5-10 minutes...\n")

search = RandomizedSearchCV(
    estimator=base_est,
    param_distributions=param_dist,
    n_iter=50,
    scoring="roc_auc",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    verbose=1,
    n_jobs=-1,
    random_state=42
)

search.fit(X_res, y_res)

# 8️⃣ Get best estimator and params
model = search.best_estimator_
print("\n🏆 Best hyperparameters found:")
for param, value in search.best_params_.items():
    print(f"   {param}: {value}")
print(f"   Best CV ROC-AUC Score: {search.best_score_:.4f}\n")

# 9️⃣ Predictions on test set
print("📊 Evaluating on test set...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 🔟 Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\n✅ Model Evaluation on Test Set:")
print(f"   Accuracy:  {accuracy * 100:.2f}%")
print(f"   F1 Score:  {f1:.4f}")
print(f"   ROC-AUC:   {roc_auc:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))
print("🔲 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 1️⃣1️⃣ Cross-validation on full dataset
print("\n📈 Running 5-Fold Cross-Validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
cv_scores_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

print(f"   5-Fold CV ROC-AUC: {np.mean(cv_scores_auc):.4f} ± {np.std(cv_scores_auc):.4f}")
print(f"   5-Fold CV Accuracy: {np.mean(cv_scores_acc):.4f} ± {np.std(cv_scores_acc):.4f}")

# 1️⃣2️⃣ Save model
model_path = "Best_Water_RandomForest_Model.pkl"
joblib.dump(model, model_path)
print(f"\n💾 Model saved as {model_path}")

# 1️⃣3️⃣ Feature importance visualization
print("\n📊 Generating feature importance chart...")
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(X.columns[sorted_idx], feature_importance[sorted_idx], color='forestgreen')
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance (Optimized)")
plt.tight_layout()
plt.savefig("feature_importance_rf.png", dpi=100, bbox_inches='tight')
plt.close()
print("   ✓ Saved as feature_importance_rf.png")

# 1️⃣4️⃣ Summary report
print("\n" + "="*60)
print("📈 FINAL SUMMARY")
print("="*60)
print(f"Target Accuracy: 91%")
print(f"Achieved Test Accuracy: {accuracy * 100:.2f}%")
print(f"Achieved Test ROC-AUC: {roc_auc:.4f}")
print(f"Cross-Val Accuracy: {np.mean(cv_scores_acc) * 100:.2f}%")
print(f"Cross-Val ROC-AUC: {np.mean(cv_scores_auc):.4f}")

if accuracy >= 0.91:
    print("\n✅ ✓ TARGET ACHIEVED!")
else:
    gap = (0.91 - accuracy) * 100
    print(f"\n⚠️  Gap to target: {gap:.2f}%")
    
print("="*60)

# 1️⃣5️⃣ Feature importance table
print("\n📊 Top 10 Important Features:")
top_features = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False).head(10)
print(top_features.to_string(index=False))
