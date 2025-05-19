# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pydeck as pdk

# Load data
file_path = r"C:\Users\Ideapad Slim 1\OneDrive\Documents\Semester 4\Data maning\dea panda numpy\covid_19_indonesia_time_series_all.csv"
df = pd.read_csv(file_path)

# Pastikan Date format datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sidebar
st.sidebar.header("Pengaturan Cluster")
n_clusters = st.sidebar.slider("Jumlah Cluster", 2, 5, 3)

# Clustering
def perform_clustering(data, n_clusters):
    features = data[['Total Cases', 'Total Deaths', 'Total Recovered', 'Population Density']].dropna()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    return clusters

# Main content
st.title("Dashboard Analisis COVID-19 Indonesia")

# Tab layout
tab1, tab2, tab3 = st.tabs(["Peta Cluster", "Tren Harian", "Ringkasan Risiko"])

with tab1:
    st.header("Peta Sebaran Cluster Wilayah")

    # Get latest data per location
    latest_data = df.sort_values('Date').groupby('Location').last().reset_index()
    latest_data['Cluster'] = perform_clustering(latest_data, n_clusters)

    # Color mapping
    cluster_colors = {
        0: [255, 0, 0],
        1: [0, 255, 0],
        2: [0, 0, 255],
        3: [255, 255, 0],
        4: [255, 0, 255]
    }

    latest_data['color'] = latest_data['Cluster'].map(cluster_colors)

    # Create map
    view_state = pdk.ViewState(
        latitude=-2.5489,
        longitude=118.0149,
        zoom=3.5
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=latest_data,
        get_position='[Longitude, Latitude]',
        get_color='color',
        get_radius=50000,
        pickable=True
    )

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=view_state,
        layers=[layer],
        tooltip={
            "html": "<b>{Location}</b><br>Total Kasus: {Total Cases}<br>Cluster: {Cluster}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }
    ))

with tab2:
    st.header("Tren Kasus Harian Nasional")

    # Date range selector
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    selected_date = st.date_input("Pilih Rentang Tanggal", [min_date, max_date], min_value=min_date, max_value=max_date)

    # Filter data
    filtered = df[(df['Date'].dt.date >= selected_date[0]) & (df['Date'].dt.date <= selected_date[1])]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(filtered['Date'], filtered['New Cases'], color='red')
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Kasus Baru")
    ax.grid(True)
    st.pyplot(fig)

with tab3:
    st.header("Ringkasan Tingkat Risiko Wilayah")

    # Calculate risk metrics
    latest_data['Risk Score'] = (latest_data['Total Deaths'] * 0.4 +
                                 latest_data['Total Cases'] * 0.3 +
                                 latest_data['Population Density'] * 0.3)

    latest_data['Risk Level'] = pd.cut(latest_data['Risk Score'],
                                       bins=3,
                                       labels=["Rendah", "Sedang", "Tinggi"])

    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Wilayah", latest_data.shape[0])
    with col2:
        st.metric("Wilayah Risiko Tinggi", latest_data[latest_data['Risk Level'] == "Tinggi"].shape[0])
    with col3:
        st.metric("Wilayah Risiko Rendah", latest_data[latest_data['Risk Level'] == "Rendah"].shape[0])

    # Show table
    st.dataframe(latest_data[['Location', 'Cluster', 'Risk Level', 'Total Cases', 'Total Deaths']].sort_values('Risk Level'),
                 height=300)
