import streamlit as st
import numpy as np
import pickle
from datetime import datetime
from tensorflow import keras
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="NYC Taxi Fare Predictor", page_icon="🚕", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #FFF8F0 0%, #FFFFFF 100%);
    }
    h1 {
        color: #FF6B35;
        font-weight: 800;
        text-align: center;
        padding-bottom: 0px;
    }
    .subtitle {
        text-align: center;
        color: #666666;
        font-size: 16px;
        margin-bottom: 30px;
    }
    div.stButton > button {
        background-color: #FF6B35;
        color: white;
        font-weight: 700;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        width: 100%;
        font-size: 16px;
    }
    div.stButton > button:hover {
        background-color: #E85A2A;
        color: white;
    }
    .fare-card {
        background-color: #FFF3EC;
        border: 2px solid #FF6B35;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
    }
    .fare-amount {
        font-size: 36px;
        font-weight: 800;
        color: #FF6B35;
    }
    .distance-text {
        color: #666666;
        font-size: 15px;
        margin-top: 8px;
    }
    .map-card {
        background-color: #FFFFFF;
        border: 2px solid #FFE0D1;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .map-heading {
        font-weight: 700;
        color: #1A1A1A;
        font-size: 18px;
        margin-bottom: 10px;
    }
    .map-legend {
        display: flex;
        gap: 16px;
        font-size: 13px;
        color: #444444;
        margin-top: 8px;
        flex-wrap: wrap;
    }
    .legend-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    .map-hint {
        color: #999999;
        font-size: 12px;
        margin-top: 6px;
    }
    </style>
""", unsafe_allow_html=True)

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


def near_airport(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, ap_lat, ap_lon):
    d1 = haversine_distance(pickup_lat, pickup_lon, ap_lat, ap_lon)
    d2 = haversine_distance(dropoff_lat, dropoff_lon, ap_lat, ap_lon)
    return (d1 < 2) or (d2 < 2)


def build_route_map(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    center_lat_m = (pickup_lat + dropoff_lat) / 2
    center_lon_m = (pickup_lon + dropoff_lon) / 2

    m = folium.Map(
        location=[center_lat_m, center_lon_m],
        zoom_start=12,
        tiles="OpenStreetMap"
    )

    folium.Marker(
        [pickup_lat, pickup_lon],
        popup="Pickup",
        icon=folium.Icon(color="orange", icon="play")
    ).add_to(m)

    folium.Marker(
        [dropoff_lat, dropoff_lon],
        popup="Drop-off",
        icon=folium.Icon(color="blue", icon="stop")
    ).add_to(m)

    folium.PolyLine(
        locations=[[pickup_lat, pickup_lon], [dropoff_lat, dropoff_lon]],
        color="#FF6B35",
        weight=4,
        opacity=0.8
    ).add_to(m)

    m.fit_bounds(
        [[pickup_lat, pickup_lon], [dropoff_lat, dropoff_lon]],
        padding=(30, 30)
    )

    return m


# --- Session state so map clicks can update the coordinate fields below ---
if "pickup_lat" not in st.session_state:
    st.session_state.pickup_lat = 40.7580
if "pickup_lon" not in st.session_state:
    st.session_state.pickup_lon = -73.9855
if "dropoff_lat" not in st.session_state:
    st.session_state.dropoff_lat = 40.6413
if "dropoff_lon" not in st.session_state:
    st.session_state.dropoff_lon = -73.7781
if "click_stage" not in st.session_state:
    st.session_state.click_stage = "pickup"


st.markdown("<h1>NYC Taxi Fare Prediction</h1>", unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Enter your trip details below to get an instant fare estimate</p>',
    unsafe_allow_html=True
)

form_col, map_col = st.columns([1, 1])

# --- Map column is processed FIRST so any session_state update from a click
# happens BEFORE the number_input widgets below are instantiated. Writing to
# st.session_state["pickup_lat"] (etc.) after those widgets exist in this run
# is what caused the StreamlitWidgetAlreadyInstantiatedError. ---
with map_col:
    with st.container(border=True):
        st.markdown('<div class="map-heading">Trip Route on Map</div>', unsafe_allow_html=True)

        route_map = build_route_map(
            st.session_state.pickup_lat,
            st.session_state.pickup_lon,
            st.session_state.dropoff_lat,
            st.session_state.dropoff_lon
        )
        map_data = st_folium(
            route_map,
            width=None,
            height=380,
            key="route_map"
        )

        st.markdown(
            '<div class="map-legend">'
            '<span><span class="legend-dot" style="background:#FF6B35;"></span>Pickup Location</span>'
            '<span><span class="legend-dot" style="background:#1E88E5;"></span>Drop-off Location</span>'
            '<span><span class="legend-dot" style="background:#FF6B35; border-radius:2px;"></span>Route</span>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="map-hint">Click the map to set the '
            f'{"pickup" if st.session_state.click_stage == "pickup" else "drop-off"} location.</div>',
            unsafe_allow_html=True
        )

        if map_data and map_data.get("last_clicked"):
            clicked_lat = map_data["last_clicked"]["lat"]
            clicked_lon = map_data["last_clicked"]["lng"]

            if st.session_state.click_stage == "pickup":
                if (clicked_lat, clicked_lon) != (st.session_state.pickup_lat, st.session_state.pickup_lon):
                    st.session_state.pickup_lat = clicked_lat
                    st.session_state.pickup_lon = clicked_lon
                    st.session_state.click_stage = "dropoff"
                    st.rerun()
            else:
                if (clicked_lat, clicked_lon) != (st.session_state.dropoff_lat, st.session_state.dropoff_lon):
                    st.session_state.dropoff_lat = clicked_lat
                    st.session_state.dropoff_lon = clicked_lon
                    st.session_state.click_stage = "pickup"
                    st.rerun()

with form_col:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Pickup Location**")
        pickup_lat = st.number_input("Pickup Latitude", format="%.6f", key="pickup_lat")
        pickup_lon = st.number_input("Pickup Longitude", format="%.6f", key="pickup_lon")
        st.markdown("**Trip Details**")
        passenger_count = st.slider("Passenger Count", 1, 6, 1)

    with col2:
        st.markdown("**Drop-off Location**")
        dropoff_lat = st.number_input("Drop-off Latitude", format="%.6f", key="dropoff_lat")
        dropoff_lon = st.number_input("Drop-off Longitude", format="%.6f", key="dropoff_lon")
        st.markdown("**Date & Time**")
        date_input = st.date_input("Date")
        time_input = st.time_input("Time")

    st.write("")
    predict_clicked = st.button("Predict Fare")

if predict_clicked:
    dt = datetime.combine(date_input, time_input)
    hour = dt.hour
    day_of_week = dt.weekday()
    month = dt.month
    is_weekend = int(day_of_week in [5, 6])
    is_rush_hour = int(
        not is_weekend and ((7 <= hour <= 10) or (16 <= hour <= 19))
    )

    trip_distance_km = haversine_distance(
        pickup_lat,
        pickup_lon,
        dropoff_lat,
        dropoff_lon
    )

    lat_diff = dropoff_lat - pickup_lat
    lon_diff = dropoff_lon - pickup_lon
    distance_per_passenger = trip_distance_km / passenger_count

    pickup_dist_from_center = haversine_distance(
        pickup_lat,
        pickup_lon,
        center_lat,
        center_lon
    )

    near_jfk = near_airport(
        pickup_lat,
        pickup_lon,
        dropoff_lat,
        dropoff_lon,
        jfk_lat,
        jfk_lon
    )

    near_lga = near_airport(
        pickup_lat,
        pickup_lon,
        dropoff_lat,
        dropoff_lon,
        lga_lat,
        lga_lon
    )

    is_airport_trip = int(near_jfk or near_lga)

    features = np.array([[
        trip_distance_km,
        lat_diff,
        lon_diff,
        passenger_count,
        hour,
        day_of_week,
        month,
        is_weekend,
        is_rush_hour,
        distance_per_passenger,
        pickup_dist_from_center,
        is_airport_trip
    ]])

    features_scaled = scaler.transform(features)
    predicted_fare = model.predict(features_scaled, verbose=0)[0][0]

    st.markdown(
        '<div class="fare-card">'
        '<div>Estimated Taxi Fare</div>'
        '<div class="fare-amount">$' + f"{predicted_fare:.2f}" + '</div>'
        '<div class="distance-text">Trip Distance: ' + f"{trip_distance_km:.2f}" + ' km</div>'
        '</div>',
        unsafe_allow_html=True
    )
