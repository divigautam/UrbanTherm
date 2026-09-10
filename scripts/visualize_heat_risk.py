import pandas as pd
import matplotlib.pyplot as plt

input_file = "data/processed/ghaziabad_heat_risk.csv"
output_file = "docs/heat_risk_trend.png"

df = pd.read_csv(input_file)
df["time"] = pd.to_datetime(df["time"])

# Plot heat-risk score over time
plt.figure(figsize=(12, 5))
plt.plot(df["time"], df["heat_risk_score"])

plt.title("Ghaziabad Daily Heat Risk - 2025")
plt.xlabel("Date")
plt.ylabel("Heat Risk Score")
plt.grid(True)

plt.tight_layout()
plt.savefig(output_file, dpi=150)
plt.show()

print("Heat-risk visualization created!")
print("Saved to:", output_file)