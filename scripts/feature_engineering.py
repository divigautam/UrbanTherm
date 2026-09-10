import pandas as pd

# Read weather data
input_file = "data/processed/ghaziabad_weather.csv"
output_file = "data/processed/ghaziabad_weather_features.csv"

df = pd.read_csv(input_file)

# Rename columns for simplicity
df = df.rename(columns={
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m": "wind_speed"
})

# Heat-related features
df["heat_index_simple"] = (
    df["temperature"] + 0.05 * df["humidity"]
)

# Handle missing values
df = df.dropna()

# Save processed data
df.to_csv(output_file, index=False)

print("Feature engineering completed!")
print(f"Rows: {len(df)}")
print(f"Saved to: {output_file}")