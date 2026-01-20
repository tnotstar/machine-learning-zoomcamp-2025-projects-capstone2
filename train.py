import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
import pickle

# --- 1. Load the prepared dataset ---
df = pd.read_parquet("prepared-stroke-prediction-dataset.parquet")

# --- 2. Define target and feature lists (as determined during EDA) ---
target_variable = "stroke"
numerical_features = ["age", "avg_glucose_level", "bmi"]
categorical_features = [
    "ever_married",
    "gender",
    "heart_disease",
    "hypertension",
    "residence_type",
    "smoking_status",
    "work_type",
]

# Combine all features for DictVectorizer
all_features = numerical_features + categorical_features

# --- 3. Split data into features (X) and target (y) ---
y = df[target_variable]
X = df[all_features]

# --- 4. Initialize and fit DictVectorizer on the full dataset ---
dv = DictVectorizer(sparse=False)
data_dict = X.to_dict(orient="records")
X_processed = dv.fit_transform(data_dict)

# --- 5. Define the best hyperparameters found during tuning ---
best_params = {
    "learning_rate": 0.01,
    "max_depth": 5,
    "n_estimators": 200,
    "subsample": 0.7,
}

# --- 6. Initialize and train the XGBoost model with best parameters ---
model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    random_state=11562788,  # Use the same random state as before
    **best_params,
)

print("Training final XGBoost model with tuned parameters...")
y_pred = model.fit(X_processed, y)
print("Model training complete.")
y_pred = model.predict_proba(X_processed)[:, 1]
roc_auc = roc_auc_score(y, y_pred)
print(f"All data AUC with best model: {roc_auc:.3f}")

# --- 7. Serialize the DictVectorizer and the trained model ---
output_file = "pipeline_v1.bin"
with open(output_file, "wb") as output:
    pickle.dump((dv, model), output)  # type: ignore
print(f"Model saved to {output_file}")

print("Deployment assets prepared successfully.")
