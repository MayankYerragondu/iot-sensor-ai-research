import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import joblib

# ----------------------------------------------------
# 1. Load dataset
# ----------------------------------------------------
df = pd.read_csv("piralarm_xgboost_training.csv")

# Features and target
features = ["hour", "minute", "day_of_week", "time_diff"]
target = "label"

X = df[features]
y = df[target]

# ----------------------------------------------------
# 2. Train / Test split
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------------------
# 3. Define XGBoost model
# ----------------------------------------------------
params = {
    "objective": "binary:logistic",   # binary classification
    "eval_metric": "auc",             # evaluation metric
    "max_depth": 5,                   # tree depth
    "learning_rate": 0.1,             # step size
    "n_estimators": 300,              # number of boosting rounds
    "subsample": 0.8,                 # random row sampling
    "colsample_bytree": 0.8,          # random feature sampling
    "random_state": 42
}

model = xgb.XGBClassifier(**params)

# ----------------------------------------------------
# 4. Train model
# ----------------------------------------------------
model.fit(X_train, y_train)

# ----------------------------------------------------
# 5. Evaluate
# ----------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("\n📊 Evaluation Results:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")
print("Confusion Matrix:\n", cm)

# ----------------------------------------------------
# 6. Feature importance
# ----------------------------------------------------
import matplotlib.pyplot as plt

xgb.plot_importance(model, importance_type="gain", show_values=False)
plt.title("Feature Importance (XGBoost - PIR Alarm)")
plt.tight_layout()
plt.show()

# ----------------------------------------------------
# 7. Save model for later use
# ----------------------------------------------------
joblib.dump(model, "piralarm_xgboost_model.joblib")
print("\n✅ Model saved to piralarm_xgboost_model.joblib")
