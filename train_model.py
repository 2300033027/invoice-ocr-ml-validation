import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("training_data.csv")

# -----------------------------
# CLEANING (IMPORTANT FIX)
# -----------------------------
df = df.dropna()

# Convert ALL labels to string (fixes float/string error)
df["vendor"] = df["vendor"].astype(str)
df["date"] = df["date"].astype(str)
df["total"] = df["total"].astype(str)

X = df["text"]
y = df[["vendor", "date", "total"]]

# -----------------------------
# MODEL (More Stable Than Logistic)
# -----------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultiOutputClassifier(
        RandomForestClassifier(n_estimators=200, random_state=42)
    ))
])

# Train
model.fit(X, y)

# Save model
joblib.dump(model, "invoice_model.pkl")

print("✅ Model trained and saved successfully!")