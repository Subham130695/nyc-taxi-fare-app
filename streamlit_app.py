
import streamlit as st
import numpy as np
import pickle
from datetime import datetime
from tensorflow import keras

st.set_page_config(page_title="NYC Taxi Fare Predictor", page_icon="🚕")

@st.cache_resource
def load_model_and_scaler():
    model = keras.models.load_model("taxi_fare_model.h5")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

jfk_lat, jfk_lon = 40.6413, -73.7781
lga_lat, lga_lon = 40.7769, -73.8740
center_lat, center_lon = 40.7831, -73.9712

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def render_map(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    map_html = f"""
    <div style="border-radius:16px; overflow:hidden; box-shadow:0 2px 10px rgba(0,0,0,0.08); border:1px solid #eee;">
      <div id="trip-map" style="height:420px; width:100%;"></div>
    </div>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"></script>
    <script>
      var map = L.map('trip-map');
      L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
          attribution: '&copy; OpenStreetMap contributors',
          maxZoom: 19
      }}).addTo(map);

      var pickup = [{pickup_lat}, {pickup_lon}];
      var dropoff = [{dropoff_lat}, {dropoff_lon}];

      var orangeIcon = L.icon({{
          iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-orange.png',
          shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
          iconSize: [25, 41], iconAnchor: [12, 41]
      }});
      var blueIcon = L.icon({{
          iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-blue.png',
          shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
          iconSize: [25, 41], iconAnchor: [12, 41]
      }});

      L.marker(pickup, {{icon: orangeIcon}}).addTo(map).bindPopup("Pickup<br>" + pickup[0] + ", " + pickup[1]);
      L.marker(dropoff, {{icon: blueIcon}}).addTo(map).bindPopup("Drop-off<br>" + dropoff[0] + ", " + dropoff[1]);

      var routeLine = L.polyline([pickup, dropoff], {{color: '#FF6B35', weight: 4, opacity: 0.8}}).addTo(map);

      var bounds = L.latLngBounds([pickup, dropoff]);
      map.fitBounds(bounds, {{padding: [40, 40]}});
    </script>
    """
    st.components.v1.html(map_html, height=440)

st.title("NYC Taxi Fare Prediction")
st.write("Enter trip details to get an estimated fare.")

col1, col2 = st.columns(2)
with col1:
    pickup_lat = st.number_input("Pickup Latitude", value=40.7580, format="%.6f")
    pickup_lon = st.number_input("Pickup Longitude", value=-73.9855, format="%.6f")
    dropoff_lat = st.number_input("Drop-off Latitude", value=40.6413, format="%.6f")
    dropoff_lon = st.number_input("Drop-off Longitude", value=-73.7781, format="%.6f")
with col2:
    passenger_count = st.slider("Passenger Count", 1, 6, 1)
    date_input = st.date_input("Date")
    time_input = st.time_input("Time")

if st.button("Predict Fare", type="primary"):
    dt = datetime.combine(date_input, time_input)
    hour = dt.hour
    day_of_week = dt.weekday()
    month = dt.month
    is_weekend = int(day_of_week in [5, 6])
    is_rush_hour = int(not is_weekend and ((7 <= hour <= 10) or (16 <= hour <= 19)))

    trip_distance_km = haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    lat_diff = dropoff_lat - pickup_lat
    lon_diff = dropoff_lon - pickup_lon
    distance_per_passenger = trip_distance_km / passenger_count
    pickup_dist_from_center = haversine_distance(pickup_lat, pickup_lon, center_lat, center_lon)

    near_jfk = (haversine_distance(pickup_lat, pickup_lon, jfk_lat, jfk_lon) < 2) or \
               (haversine_distance(dropoff_lat, dropoff_lon, jfk_lat, jfk_lon) < 2)
    near_lga = (haversine_distance(pickup_lat, pickup_lon, lga_lat, lga_lon) < 2) or \
               (haversine_distance(dropoff_lat, dropoff_lon, lga_lat, lga_lon) < 2)
    is_airport_trip = int(near_jfk or near_lga)

    features = np.array([[
        trip_distance_km, lat_diff, lon_diff, passenger_count,
        hour, day_of_week, month, is_weekend, is_rush_hour,
        distance_per_passenger, pickup_dist_from_center, is_airport_trip
    ]])

    features_scaled = scaler.transform(features)
    predicted_fare = model.predict(features_scaled, verbose=0)[0][0]

    st.success(f"Estimated Taxi Fare: ${predicted_fare:.2f}")
    st.info(f"Trip Distance: {trip_distance_km:.2f} km")

# --- TEMPORARY: testing the map renders correctly (Step 1 only) ---
render_map(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
