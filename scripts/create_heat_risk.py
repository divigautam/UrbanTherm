import pandas as pd

input_file = "data/processed/ghaziabad_weather_features.csv"
output_file = "data/processed/ghaziabad_heat_risk.csv"

df = pd.read_csv(input_file)
df["time"] = pd.to_datetime(df["time"])

# Daily weather summary
daily = df.groupby(df["time"].dt.date).agg(
    avg_temperature=("temperature", "mean"),
    max_temperature=("temperature", "max"),
    avg_humidity=("humidity", "mean"),
    avg_wind_speed=("wind_speed", "mean"),
    avg_heat_index=("heat_index_simple", "mean")
).reset_index()

# Simple prototype heat-risk score
daily["heat_risk_score"] = (
    0.6 * daily["avg_heat_index"]
    + 0.3 * daily["max_temperature"]
    + 0.1 * daily["avg_humidity"]
)

# Low / Medium / High based on historical distribution
low_limit = daily["heat_risk_score"].quantile(0.33)
high_limit = daily["heat_risk_score"].quantile(0.66)

def classify(score):
    if score <= low_limit:
        return "Low"
    elif score <= high_limit:
        return "Medium"
    else:
        return "High"

daily["heat_risk"] = daily["heat_risk_score"].apply(classify)

daily.to_csv(output_file, index=False)

print("Heat-risk dataset created!")
print(daily["heat_risk"].value_counts())
print("Saved to:", output_file)