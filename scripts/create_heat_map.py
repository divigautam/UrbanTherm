import pandas as pd
import folium

input_file = "data/processed/ghaziabad_real_heat_risk.csv"

df = pd.read_csv(input_file)

center_lat = df["centroid_lat"].mean()
center_lon = df["centroid_lon"].mean()


def create_map(risk_filter, filename):

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11
    )

    if risk_filter == "All":
        data = df
    else:
        data = df[df["heat_risk"] == risk_filter]

    for _, row in data.iterrows():

        if row["heat_risk"] == "High":
            color = "red"
        elif row["heat_risk"] == "Medium":
            color = "orange"
        else:
            color = "green"

        popup = f"""
        <b>Grid:</b> {row['grid_id']}<br>
        <b>Risk:</b> {row['heat_risk']}<br>
        <b>LST:</b> {row['lst_celsius']:.2f} °C<br>
        <b>NDVI:</b> {row['ndvi']:.3f}<br>
        <b>Risk Score:</b> {row['heat_risk_score']:.1f}
        """

        folium.CircleMarker(
            location=[
                row["centroid_lat"],
                row["centroid_lon"]
            ],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            popup=popup
        ).add_to(m)

    m.save(f"docs/{filename}")

    print(f"{risk_filter} map created!")


create_map("All", "ghaziabad_heat_risk_map.html")
create_map("High", "ghaziabad_high_risk_map.html")
create_map("Medium", "ghaziabad_medium_risk_map.html")
create_map("Low", "ghaziabad_low_risk_map.html")

print("REAL SATELLITE HEAT MAPS CREATED!")