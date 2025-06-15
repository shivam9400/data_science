import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import statsmodels.api as sm

st.set_page_config(page_title="Strava Dashboard", layout="wide")


# --------------- Sidebar ---------------------
st.sidebar.title("📂 Data Source")
data_source = st.sidebar.radio("Choose data source:", ["Use GitHub Sample", "Upload CSV"])
@st.cache_data

def load_sample_data():
    return pd.read_csv(st.secrets["GITHUB_CSV_URL"])

df = None
if data_source == "Use GitHub Sample":
    df = load_sample_data()
elif data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your Strava CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

if df is None:
    st.warning("Please select a data source.")
    st.stop()
# --------------- Main Layout -----------------
st.title("🏃‍♂️ Strava Running Dashboard")

if len(df)>0:
    # Transform and clean data
    df['start_date'] = pd.to_datetime(df['start_date']) # start_date as datetime object
    df['distance_km'] = df['distance'] / 1000  # meters to km
    df['moving_time_min'] = df['moving_time'] / 60  # seconds to minutes
    df['elapsed_time_min'] = df['elapsed_time'] / 60  # seconds to minutes
    df['pace_min_per_km'] = df['moving_time_min'] / df['distance_km']

    # Filter by sport_type and other data processing
    runs = df[df['type'] == 'Run'].copy()
    runs = runs[((runs['pace_min_per_km']<30) & 
                (runs['pace_min_per_km']>=4))  # Filter out extreme paces
                ]
    
    runs['training_load'] = runs['distance_km']      # Add a training load proxy
    runs['week'] = runs['start_date'].dt.to_period('W').apply(lambda r: r.start_time)

    # --- SECTION 1: Summary Stats ---
    with st.expander("Summary Statistics", expanded=False):
        st.write(runs[['start_date', 'distance_km', 'pace_min_per_km']].describe())
    
    # --- SECTION 2: Box plot of Pace ---
    with st.expander("Pace Distribution (Box Plot)"):
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.boxplot(x=runs['pace_min_per_km'], ax=ax1)
        ax1.set_title("Pace (min/km)")
        st.pyplot(fig1)

    # --- SECTION 3: Distance vs Time ---
    with st.expander("Distance vs Time"):
        fig2 = px.line(runs, x='start_date', y='distance_km',
                          title="Distance (km) over Time", 
                          labels={"start_date": "Date", "distance_km": "Distance (km)"}
                          )
        fig2.update_traces(line=dict(color='teal'))
        st.plotly_chart(fig2, use_container_width=True)
    
    # --- SECTION 4: Pace vs Time ---
    with st.expander("Pace vs Time"):
        fig3 = px.line(runs, x='start_date', y='pace_min_per_km',
                       title="Pace over Time", 
                       labels={"start_date": "Date", "pace_min_per_km": "Pace (min/km)"}
                       )
        fig3.update_traces(line=dict(color='orange'))
        fig3.update_yaxes(autorange='reversed')
        st.plotly_chart(fig3, use_container_width=True)

    # --- SECTION 5: Weekly Running Volume ---
    with st.expander("Weekly Running Volume"):
        weekly = runs.groupby('week')['distance_km'].sum().reset_index()
        fig4 = px.line(weekly, x='week', y='distance_km', 
                       title="Weekly Running Volume",
                       labels={"week": "Week", "distance_km": "Distance (km)"},
                       markers=True
                       )
        fig4.update_traces(line=dict(color='purple'))
        st.plotly_chart(fig4, use_container_width=True)

    # --- SECTION 6: CTL / ATL / TSB ---
    with st.expander("Fitness Metrics: CTL, ATL, TSB"):
        daily_load = runs.groupby(runs['start_date'].dt.date).agg({'training_load': 
                                                           'sum'}).reset_index()
        daily_load['start_date'] = pd.to_datetime(daily_load['start_date'])
        date_range = pd.date_range(start=daily_load['start_date'].min(), 
                                   end=daily_load['start_date'].max())
        daily_load = daily_load.set_index('start_date').reindex(date_range).fillna(0.0)
        daily_load.index.name = 'date'
        daily_load = daily_load.rename(columns={'training_load': 'load'}).reset_index()

        daily_load['CTL'] = daily_load['load'].rolling(window=42, min_periods=1).mean()
        daily_load['ATL'] = daily_load['load'].rolling(window=7, min_periods=1).mean()
        daily_load['TSB'] = daily_load['CTL'] - daily_load['ATL']

        fig5 = px.line(daily_load, 
                       x='date', y=['CTL', 'ATL', 'TSB'],
                       title="Fitness Metrics Over Time",
                       labels={
                           "value": "Score",
                           "date": "Date",
                           "variable": "Metric"
                           },
                           color_discrete_map={
                               'CTL': 'green',
                               'ATL': 'red',
                               'TSB': 'blue'}
                               )
        st.plotly_chart(fig5, use_container_width=True)

    # --- SECTION 7: Fatigue vs Pace ---
    with st.expander("Fatigue (TSB) vs Performance (Pace)"):
        daily_load['date'] = pd.to_datetime(daily_load['date']).dt.floor('D').dt.tz_localize(None)
        runs['start_date'] = pd.to_datetime(runs['start_date']).dt.floor('D').dt.tz_localize(None)

        merged = pd.merge(runs, daily_load[['date', 'TSB']], 
                          left_on=runs['start_date'],
                          right_on=daily_load['date'], how='left')
        
        fig6 = px.scatter(merged, x='TSB', y='pace_min_per_km',
                          trendline='ols', title="Fatigue (TSB) vs Performance (Pace)")
        fig6.update_traces(selector=dict(mode='lines'), line=dict(color='red')) 
        st.plotly_chart(fig6, use_container_width=True)

    # --- SECTION 8: Clustering Pace & Distance ---
    with st.expander("Activity Clustering (Pace vs Distance)"):
        cluster_data = runs[['distance_km', 'pace_min_per_km']].dropna()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(cluster_data)

        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(scaled)
        cluster_data['cluster'] = labels

        fig7 = px.scatter(cluster_data, x='distance_km', y='pace_min_per_km',
                          color=cluster_data['cluster'].astype(str), title="Pace & Distance Clustering")
        st.plotly_chart(fig7, use_container_width=True)

    # --- SECTION 9: Heatmap of Training Load ---
    with st.expander("Heatmap of Training Load"):
        calendar_df = runs[['start_date', 'training_load']].copy()
        calendar_df['week'] = calendar_df['start_date'].dt.to_period('W').apply(lambda r: r.start_time.date())
        calendar_df['dow'] = calendar_df['start_date'].dt.dayofweek
        dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        calendar_df['dow'] = calendar_df['dow'].apply(lambda x: dow_labels[x])
        calendar_df['dow'] = pd.Categorical(calendar_df['dow'], categories=dow_labels, ordered=True)
        grouped = calendar_df.groupby(['dow', 'week'])['training_load'].sum().reset_index()
        pivot = grouped.pivot(index='dow', columns='week', values='training_load')
        mask = pivot == 0

        fig8, ax8 = plt.subplots(figsize=(20, 8))
        sns.heatmap(pivot, mask=mask, cmap="YlOrRd", annot=True, 
                    fmt=".0f", linewidths=0.5, linecolor='gray', 
                    cbar_kws={'label': 'Training Load'},
                    square=False,
                    )
        plt.title("Training Load Heatmap by Week (Mon–Sun)")
        plt.xlabel("Week Starting")
        plt.ylabel("Day of Week")
        #plt.xticks(rotation=45)
        st.pyplot(fig8)

    # --- SECTION 10: Training Feedback ---
    with st.expander("Automatic Training Feedback"):
        recent = daily_load.tail(7)['load']
        avg_load = recent.mean()
        tsb_now = daily_load.iloc[-1]['TSB']
        pace_recent = runs[runs['start_date'] > runs['start_date'].max() - pd.Timedelta(days=7)]['pace_min_per_km'].mean()

        st.markdown(f"**🧠 Avg Load (Last 7 days):** {avg_load:.2f}")
        st.markdown(f"**📉 Current Fatigue (TSB):** {tsb_now:.2f}")
        st.markdown(f"**🏃 Avg Pace (Last 7 days):** {pace_recent:.2f} min/km")

        if tsb_now < -20:
            st.warning("You're likely fatigued. Consider reducing training load.")
        elif tsb_now > 20:
            st.info("You're fresh. It's a good time to increase intensity.")
        else:
            st.success("You're in a balanced state. Keep it up!")
else:
    st.info("Please upload your exported Strava activity CSV to begin.")
