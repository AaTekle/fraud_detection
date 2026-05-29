# src/predict.py
from pathlib import Path
import joblib
import pandas as pd

# Define the saved model path and input dataset path.
MODEL_PATH = Path("models/isolation_forest.joblib")
DATA_PATH = Path("data/creditcard.csv")

# Load the trained model and credit card dataset.
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# Prepare features by removing the target label column.
X = df.drop(columns=["Class"])

# Generate anomaly scores, with higher values meaning more suspicious.
scores = -model.decision_function(X)

# Copy the original data, attach anomaly scores, and rank transactions by suspicion.
out = df.copy()
out["anomaly_score"] = scores
out = out.sort_values("anomaly_score", ascending=False)

# Save the top 100 most suspicious transactions to the reports folder.
out.head(100).to_csv("reports/top_100_suspicious_transactions.csv", index=False)

# Print the top 20 suspicious transactions with key fields.
print(out[["Time", "Amount", "Class", "anomaly_score"]].head(20))