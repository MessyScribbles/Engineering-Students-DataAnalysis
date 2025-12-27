# train_model.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# =========================
# Load Dataset ( make sure to adjust the path accordingly )
# =========================
print(">>> Script started")

df = pd.read_csv(r"D:\Downloads\EngStudent_Db(Eng_Stats).csv", sep=";")

# Cleaning
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].str.replace(",", ".", regex=False)

df["feeling_lost"] = df["feeling_lost"].map({"Yes": 1, "No": 0})
df["Schedule_effect"] = df["Schedule_effect"].map({"Yes": 1, "No": 0})

study_map = {
    "Daily": 0,
    "Few times during the week": 1,
    "Day before exam": 2
}
df["Studying_Consistency_Encoded"] = df["Studying_Consistency"].map(study_map)
df["Gender_Encoded"] = df["Gender"].map({"M": 0, "F": 1})

numeric_cols = [
    "Stress_level",
    "Burnout_Score",
    "Average_hours_sleep",
    "Class attendance",
    "Troubled_Modules",
    "Performance_score"
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

features = [
    "Stress_level",
    "Burnout_Score",
    "Average_hours_sleep",
    "Class attendance",
    "Studying_Consistency_Encoded",
    "feeling_lost",
    "Schedule_effect",
    "Troubled_Modules",
    "Gender_Encoded"
]

X = df[features]
y = df["Performance_score"]

mask = y.notna()
X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# =========================
# Decision Tree Model (feel free to adjust hyperparameters)
# =========================
model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("tree", DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=8,
        random_state=42
    ))
])

model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# Save model + accuracy
joblib.dump(
    {"model": model, "accuracy": accuracy},
    "model_tree.pkl"
)

print("Model saved with accuracy:", accuracy)
