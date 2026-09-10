import pandas as pd

weather_file = "data/processed/ghaziabad_weather_features.csv"
satellite_file = "data/processed/ghaziabad_satellite_features.csv"
output_file = "data/processed/ghaziabad_final_features.csv"

# Load weather
weather = pd.read_csv(weather_file)
weather["time"] = pd.to_datetime(weather["time"])

# Select satellite date
date = "2025-06-07"

day = weather[
    weather["time"].dt.strftime("%Y-%m-%d") == date
].copy()

# Daily weather summary
daily_weather = pd.DataFrame({
    "date": [date],
    "avg_temperature": [day["temperature"].mean()],
    "max_temperature": [day["temperature"].max()],
    "avg_humidity": [day["humidity"].mean()],
    "avg_wind_speed": [day["wind_speed"].mean()],
    "avg_heat_index": [day["heat_index_simple"].mean()]
})

# Load satellite grid data
satellite = pd.read_csv(satellite_file)

# Add same date to every grid cell
satellite["date"] = date

# Add daily weather values to every grid cell
weather_values = daily_weather.iloc[0]

for column in [
    "avg_temperature",
    "max_temperature",
    "avg_humidity",
    "avg_wind_speed",
    "avg_heat_index"
]:
    satellite[column] = weather_values[column]

# Save final dataset
satellite.to_csv(output_file, index=False)

print("Final feature dataset created!")
print("Rows:", len(satellite))
print("Columns:", satellite.columns.tolist())
print("\nSample:")
print(satellite.head())
print("\nSaved:", output_file)