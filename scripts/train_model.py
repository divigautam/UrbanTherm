import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

input_file = "data/processed/ghaziabad_real_heat_risk.csv"

df = pd.read_csv(input_file)

features = [
    "ndvi",
    "ndbi",
    "avg_temperature",
    "max_temperature",
    "avg_humidity",
    "avg_wind_speed",
    "avg_heat_index"
]

X = df[features]
y = df["heat_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n================================")
print("URBANTHERM ML MODEL")
print("================================")
print("Training samples:", len(df))
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nFeature Importance:")

for feature, importance in zip(features, model.feature_importances_):
    print(feature, ":", round(importance, 4))

joblib.dump(model, "ml/heat_risk_model.pkl")

print("\nModel saved to: ml/heat_risk_model.pkl")