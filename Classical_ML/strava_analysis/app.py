import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit page config
st.set_page_config(page_title="Strava Activity Analysis", layout="centered")

# App title
st.title("🏃‍♂️ Strava Activity Dashboard")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your Strava activity CSV file", type=["csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)

    # Basic sanity check
    if 'distance' not in df.columns or 'moving_time' not in df.columns:
        st.error("❌ The uploaded CSV does not have required 'distance' and 'moving_time' columns.")
    else:
        # Calculate pace (min/km)
        df = df[df['distance'] > 0]  # Avoid divide-by-zero
        df['distance_km'] = df['distance'] / 1000
        df['moving_time_min'] = df['moving_time'] / 60
        df['pace_min_per_km'] = df['moving_time_min'] / df['distance_km']

        # Remove outliers using IQR
        Q1 = df['pace_min_per_km'].quantile(0.25)
        Q3 = df['pace_min_per_km'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean = df[(df['pace_min_per_km'] >= lower) & (df['pace_min_per_km'] <= upper)]

        # Show basic stats
        st.subheader("📊 Summary Statistics (Pace min/km)")
        st.write(df_clean['pace_min_per_km'].describe())

        # Box plot
        st.subheader("📦 Pace Distribution (Outliers Removed)")
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.boxplot(x=df_clean['pace_min_per_km'], color="skyblue", ax=ax)
        ax.set_title("Pace (min/km) Box Plot")
        st.pyplot(fig)

        # Optional: Display table
        if st.checkbox("Show cleaned data table"):
            st.dataframe(df_clean[['name', 'start_date', 'pace_min_per_km']].sort_values(by='start_date', ascending=False))
else:
    st.info("Please upload your exported Strava activity CSV to begin.")
