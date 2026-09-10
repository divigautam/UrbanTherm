import pandas as pd

input_file = "data/processed/ghaziabad_final_features.csv"
output_file = "data/processed/ghaziabad_real_heat_risk.csv"

df = pd.read_csv(input_file)

# LST-based thresholds
low_threshold = df["lst_celsius"].quantile(0.33)
high_threshold = df["lst_celsius"].quantile(0.66)

def classify(lst):
    if lst <= low_threshold:
        return "Low"
    elif lst <= high_threshold:
        return "Medium"
    else:
        return "High"

df["heat_risk"] = df["lst_celsius"].apply(classify)

# A normalized 0-100 heat score
lst_min = df["lst_celsius"].min()
lst_max = df["lst_celsius"].max()

df["heat_risk_score"] = (
    (df["lst_celsius"] - lst_min)
    / (lst_max - lst_min)
    * 100
)

df.to_csv(output_file, index=False)

print("REAL HEAT-RISK DATASET CREATED!")
print("--------------------------------")
print("Low threshold:", low_threshold, "°C")
print("High threshold:", high_threshold, "°C")

print("\nRisk distribution:")
print(df["heat_risk"].value_counts())

print("\nSaved:", output_file)