import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime
import random
import requests

st.set_page_config(page_title="Strava Auth", layout="wide")

# Restore client_id and client_secret from URL if available (after redirect)
query_params = st.query_params
if "client_id" in query_params:
    st.session_state["client_id"] = query_params["client_id"]
if "client_secret" in query_params:
    st.session_state["client_secret"] = query_params["client_secret"]
    
#st.set_page_config(page_title="Strava Dashboard", layout="wide")

hide_menu_style = """
<style>
/* Hide sidebar */
div[data-testid="stSidebar"] {
    display: none;
}

/* Expand main content area */
div[data-testid="stAppViewContainer"] > div:first-child {
    margin-left: 0rem !important;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Hide Streamlit top menu and footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Reduce padding above title */
div.block-container {
    padding-top: 1rem !important;
}

/* Dashboard button styling */
.dashboard-button {
    width: 100%;
    height: 80px;
    border: 1px solid #ccc;
    border-radius: 8px;
    background-color: white;
    font-size: 15px;
    font-weight: 500;
    color: #333;
    text-align: center;
    line-height: normal;
    padding: 10px;
    transition: all 0.2s ease-in-out;
}
.dashboard-button:hover {
    background-color: #f0f0f0;
    border-color: #999;
    transform: translateY(-2px);
    cursor: pointer;
}
</style>
"""

st.markdown(hide_menu_style, unsafe_allow_html=True)
# --- Initialize session state ---
for key in ["client_id", "client_secret"]:
    if key not in st.session_state:
        st.session_state[key] = ""

redirect_uri = "https://strava-shivamsharma.streamlit.app"
# --------------- Sidebar ---------------------
st.sidebar.title("📂 Data Source")
# data_source = st.sidebar.radio("Choose data source:", ["Use Sample", 
#                                                        "Upload CSV",
#                                                        "Strava Authentication"])
# Add this once at the top of your app
if "data_source_radio" not in st.session_state:
    st.session_state["data_source_radio"] = "Strava Authentication"

# Show radio input in sidebar
data_source = st.sidebar.radio(
    "Choose data source:",
    ["Use Sample", "Upload CSV", "Strava Authentication"],
    key="data_source_radio"
)

# Assign selected data source consistently
current_data_source = st.session_state["data_source_radio"]

# if "data_source" not in st.session_state:
#     st.session_state["data_source"] = "Strava Authentication"

# data_source = st.sidebar.radio(
#     "Choose data source:",
#     ["Use Sample", "Upload CSV", "Strava Authentication"],
#     index=["Use Sample", "Upload CSV", "Strava Authentication"].index(st.session_state["data_source"]),
#     key="data_source_radio"
# )
# st.session_state["data_source"] = data_source

@st.cache_data
def load_sample_data():
    csv_url = st.secrets["strava"]["sample_csv"]
    return pd.read_csv(csv_url)

df = None
# Reset df only when data source changes
if "previous_data_source" not in st.session_state:
    st.session_state["previous_data_source"] = current_data_source

# If the source changed, reset df
if current_data_source != st.session_state["previous_data_source"]:
    st.session_state["df"] = None
    st.session_state["previous_data_source"] = current_data_source

if current_data_source == "Strava Authentication":
    st.sidebar.markdown("🔑 Authenticate with Strava")
    st.sidebar.markdown("To connect your Strava account, enter the following details:")
        
    client_id_input = st.sidebar.text_input("Client ID", 
                                            value=st.session_state.get("client_id", ""))
    client_secret_input = st.sidebar.text_input("Client Secret", 
                                                type="password", 
                                                value=st.session_state.get("client_secret", ""))

    # Save to session state if not already saved
    if client_id_input:
        st.session_state["client_id"] = client_id_input
    if client_secret_input:
        st.session_state["client_secret"] = client_secret_input
    
    # client_id = st.text_input("Client ID")
    # client_secret = st.text_input("Client Secret", type="password")
    client_link = f"https://www.strava.com/settings/api"
    redirect_uri = "https://strava-shivamsharma.streamlit.app"
    # redirect_uri = "http://localhost:8501"

    st.sidebar.markdown(f"[Click here to get client ID and secret from Strava →]({client_link})")
    
    # Build authorization link
    # --- Generate auth link if client_id is available ---
    if st.session_state.client_id:
        redirect_uri_with_params = (
            f"{redirect_uri}?client_id={st.session_state.client_id}"
            f"&client_secret={st.session_state.client_secret}"
            )
        auth_url = (
            f"https://www.strava.com/oauth/authorize"
            f"?client_id={st.session_state.client_id}"
            f"&response_type=code"
            f"&redirect_uri={redirect_uri_with_params}"
            f"&approval_prompt=force"
            f"&scope=read,activity:read_all"
            )
        #st.markdown(f"[🚴 Click to authorize with Strava →]({auth_url})")
        # st.sidebar.markdown(
        #     f"""<a href="{auth_url}" target="_self">🚴 Click to authorize with Strava →</a>""",
        #     unsafe_allow_html=True
        #     )
        st.sidebar.markdown(
            f"""<a href="{auth_url}" target="_blank">🚴 Click to authorize with Strava →</a>""",
            unsafe_allow_html=True
            )

    
    # --- Get the code from redirect URL ---
    query_params = st.query_params
    code = query_params.get("code")

    auth_code = code or st.sidebar.text_input("Paste the `code` here")

    # --- If redirected back with code ---
    if st.sidebar.button("Get My Strava Data"):
        if not all([st.session_state.client_id, st.session_state.client_secret, auth_code]):
            st.sidebar.error("⚠️ Please fill in all required fields.")
        else:
            import requests
            token_url = "https://www.strava.com/oauth/token"
            data = {
                "client_id": st.session_state.client_id,
                "client_secret": st.session_state.client_secret,
                "code": auth_code,
                "grant_type": "authorization_code"
            }
            response = requests.post(token_url, data=data)

            if response.status_code == 200:
                tokens = response.json()
                access_token = tokens["access_token"]
                st.session_state['strava_access_token'] = access_token
                st.success("✅ Authentication successful!")

                def fetch_all_activities(token, max_pages=10):
                    all_activities = []
                    headers = {"Authorization": f"Bearer {token}"}
                    for page in range(1, max_pages + 1):
                        r = requests.get(
                            "https://www.strava.com/api/v3/athlete/activities",
                            headers=headers,
                            params={"per_page": 200, "page": page}
                        )
                        if r.status_code != 200:
                            st.error(f"❌ Error fetching page {page}: {r.status_code}")
                            break
                        batch = r.json()
                        if not batch:
                            break
                        all_activities.extend(batch)
                    return pd.DataFrame(all_activities)

                df = fetch_all_activities(access_token)
                st.session_state["df"] = df
                #st.write(df.head())

            else:
                st.sidebar.error("❌ Authentication failed. Check credentials and code.")
elif current_data_source == "Use Sample":
    df = load_sample_data()
    st.session_state["df"] = df
elif current_data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your Strava CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df

df = st.session_state.get("df", None)
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
    runs['duration_hr'] = runs['moving_time'] / 3600
    runs['month'] = runs['start_date'].dt.strftime('%b').str.upper()
    runs['month_num'] = runs['start_date'].dt.month
    runs['year'] = runs['start_date'].dt.year
    runs['week_num'] = runs['start_date'].dt.isocalendar().week

    daily_load = runs.groupby(runs['start_date'].dt.date).agg({'training_load': 'sum'}).reset_index()
    daily_load['start_date'] = pd.to_datetime(daily_load['start_date'])
    date_range = pd.date_range(start=daily_load['start_date'].min(), end=daily_load['start_date'].max())
    daily_load = daily_load.set_index('start_date').reindex(date_range).fillna(0.0)
    daily_load.index.name = 'date'
    daily_load = daily_load.rename(columns={'training_load': 'load'}).reset_index()

    daily_load['CTL'] = daily_load['load'].rolling(window=42, min_periods=1).mean()
    daily_load['ATL'] = daily_load['load'].rolling(window=7, min_periods=1).mean()
    daily_load['TSB'] = daily_load['CTL'] - daily_load['ATL']

    ##################################################################################
    
    # --- HEADER SUMMARY SECTION ---
    st.markdown("## 🗓️ Training Calendar")
    # Filter for year
    colors = ['darkorange', 'green', 'blue', 'black']
    random_color = random.choice(colors)
    current_year = datetime.today().year
    selected_year_header = st.selectbox("Select Year", sorted(runs['year'].unique()), 
                                        index=sorted(runs['year'].unique()).index(current_year) if current_year in runs['year'].unique() 
                                        else 0, 
                                        key="calendar_year")

    header_runs = runs[runs['year'] == selected_year_header].copy()
    header_runs['week_label'] = header_runs['start_date'].dt.strftime('%b') + ' - W' + header_runs['start_date'].dt.isocalendar().week.astype(str)
    header_runs['week_start'] = header_runs['start_date'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly_summary = (
        header_runs.groupby(['week_start'])['duration_hr'].sum().reset_index().sort_values('week_start')
        )
    weekly_summary['week_label'] = weekly_summary['week_start'].dt.strftime('%b - W%V')  # ISO week number

    # Stats
    total_hours = header_runs['duration_hr'].sum()
    total_km = header_runs['distance_km'].sum()
    total_activities = header_runs.shape[0]
    total_prs = header_runs['achievement_count'].fillna(0).astype(int).sum() if 'achievement_count' in header_runs.columns else 0

    # Layout
    col_calendar, col_stats = st.columns([3, 2])
    with col_calendar:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=weekly_summary['week_label'],
            y=weekly_summary['duration_hr'],
            marker_color=random_color,
            marker_line_width=0,
            hovertemplate='%{x}<br>%{y:.1f} hrs<extra></extra>'
        ))
        fig_bar.update_layout(
            height=160,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showticklabels=True, tickangle=-45),
            yaxis=dict(showticklabels=False),
            plot_bgcolor='white',
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_stats:
        col1, col2, col3 = st.columns(3)
        col1.metric("Hours", f"{int(total_hours)}")
        col2.metric("Kilometers", f"{total_km:.1f}")
        col3.metric("Activities", f"{total_activities}")

    #################################################################################
    # --- Chart Dashboard Blocks ---
    st.markdown("## 📊 Select from drop-down to View Detailed Chart")
    st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)

    chart_blocks = {
        "Summary Statistics": "summary_stats",
        "Distance vs Time": "distance_vs_time",
        "Pace vs Time": "pace_vs_time",
        "Weekly Running Volume": "weekly_volume",
        "Training Load + CTL / ATL / TSB": "ctl_atl_tsb",
        "Fatigue vs Pace": "fatigue_vs_pace",
        "Clustering Pace & Distance": "clustering_pace_distance",
        "Heatmap of Training Load": "heatmap_training_load",
        "Training Feedback": "training_feedback",
        "Training Calendar": "training_calendar",
        "Interactive Calendar View": "interactive_calendar_view"
    }

    # block_keys = list(chart_blocks.keys())
    # for i in range(0, len(block_keys), 6):
    #     cols = st.columns(6)
    #     for j in range(6):
    #         if i + j < len(block_keys):
    #             label = block_keys[i + j]
    #             with cols[j]:
    #                 if st.button(label, key=chart_blocks[label]):
    #                     st.session_state.selected_block = chart_blocks[label]
    #                 st.markdown(
    #                     f"""<style>
    #                         div.stButton > button {{
    #                             width: 100% !important;
    #                             height: 80px !important;
    #                             border-radius: 8px;
    #                             border: 1px solid #ccc;
    #                             font-weight: 500;
    #                             font-size: 15px;
    #                         }}
    #                     </style>""", unsafe_allow_html=True)

    # # Update selected block
    # selected_block = st.session_state.get("selected_block", None)

    selected_label = st.selectbox("Choose Chart:", list(chart_blocks.keys()))
    selected_block = chart_blocks[selected_label]
    st.markdown("---")
    ########################################################################################
    # --- Summary Stats ---
    if selected_block == "summary_stats":
    #with st.expander("Summary Statistics", expanded=False):
        st.write(runs[['start_date', 'distance_km', 'pace_min_per_km']].describe())
        fig = px.box(runs,
                     x='pace_min_per_km',
                    points="all",  # show all data points (optional)
                    title="Pace Distribution (min/km)",
                    labels={'pace_min_per_km': 'Pace (min/km)'},
                    template="simple_white",
                    width=900,
                    height=400 
                    )
        fig.update_traces(marker=dict(opacity=0.5, size=4))  # Optional tweaks
        st.plotly_chart(fig, use_container_width=False)
        # fig1, ax1 = plt.subplots(figsize=(10, 5))
        # sns.boxplot(x=runs['pace_min_per_km'], ax=ax1)
        # ax1.set_title("Pace (min/km)")
        # st.pyplot(fig1)

    # --- Distance vs Time ---
    elif selected_block == "distance_vs_time":
    #with st.expander("Distance vs Time"):
        fig2 = px.line(runs, x='start_date', y='distance_km',
                          title="Distance (km) over Time", 
                          labels={"start_date": "Date", "distance_km": "Distance (km)"}
                          )
        fig2.update_traces(line=dict(color='teal'))
        st.plotly_chart(fig2, use_container_width=True)
    
    # --- Pace vs Time ---
    elif selected_block == "pace_vs_time":
    #with st.expander("Pace vs Time"):
        fig3 = px.line(runs, x='start_date', y='pace_min_per_km',
                       title="Pace over Time", 
                       labels={"start_date": "Date", "pace_min_per_km": "Pace (min/km)"}
                       )
        fig3.update_traces(line=dict(color='orange'))
        fig3.update_yaxes(autorange='reversed')
        st.plotly_chart(fig3, use_container_width=True)

    # --- Weekly Running Volume ---
    elif selected_block == "weekly_volume":
    #with st.expander("Weekly Running Volume"):
        weekly = runs.groupby('week')['distance_km'].sum().reset_index()
        fig4 = px.line(weekly, x='week', y='distance_km', 
                       title="Weekly Running Volume",
                       labels={"week": "Week", "distance_km": "Distance (km)"},
                       markers=True
                       )
        fig4.update_traces(line=dict(color='purple'))
        st.plotly_chart(fig4, use_container_width=True)

    # --- Training Load + CTL / ATL / TSB ---
    elif selected_block == "ctl_atl_tsb":
    #with st.expander("Fitness, Fatigue, and Form Over Time"):
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=daily_load['date'], y=daily_load['CTL'], mode='lines', name='Fitness (CTL)', line=dict(color='green', width=2)))
        fig5.add_trace(go.Scatter(x=daily_load['date'], y=daily_load['ATL'], mode='lines', name='Fatigue (ATL)', line=dict(color='red', width=2)))
        fig5.add_trace(go.Scatter(x=daily_load['date'], y=daily_load['TSB'], mode='lines', name='Form (TSB)', line=dict(color='blue', width=2)))
        fig5.add_trace(go.Bar(x=daily_load['date'], y=daily_load['load'], name='Training Load', marker=dict(color='darkgray'), opacity=0.5, yaxis='y2'))

        fig5.update_layout(
            title="Fitness, Fatigue, and Form Over Time",
            xaxis_title="Date",
            yaxis_title="Score (proxy via distance)",
            legend_title="Metric",
            template='plotly_white',
            height=500,
            bargap=0,
            hovermode='x unified',
            yaxis=dict(title='Score', side='left'),
            yaxis2=dict(overlaying='y', side='right', showgrid=False, visible=False),
            xaxis=dict(rangeslider=dict(visible=True), type='date')
        )
        st.plotly_chart(fig5, use_container_width=True)

    # --- Fatigue vs Pace ---
    elif selected_block == "fatigue_vs_pace":
    #with st.expander("Fatigue (TSB) vs Performance (Pace)"):
        daily_load['date'] = pd.to_datetime(daily_load['date']).dt.floor('D').dt.tz_localize(None)
        runs['start_date'] = pd.to_datetime(runs['start_date']).dt.floor('D').dt.tz_localize(None)

        merged = pd.merge(runs, daily_load[['date', 'TSB']], 
                          left_on=runs['start_date'],
                          right_on=daily_load['date'], how='left')
        
        fig6 = px.scatter(merged, x='TSB', y='pace_min_per_km',
                          trendline='ols', title="Fatigue (TSB) vs Performance (Pace)")
        fig6.update_traces(selector=dict(mode='lines'), line=dict(color='red')) 
        st.plotly_chart(fig6, use_container_width=True)

    # --- Clustering Pace & Distance ---
    elif selected_block == "clustering_pace_distance":
    #with st.expander("Activity Clustering (Pace vs Distance)"):
        cluster_data = runs[['distance_km', 'pace_min_per_km']].dropna()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(cluster_data)

        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(scaled)
        cluster_data['cluster'] = labels

        fig7 = px.scatter(cluster_data, x='distance_km', y='pace_min_per_km',
                          color=cluster_data['cluster'].astype(str), title="Pace & Distance Clustering")
        st.plotly_chart(fig7, use_container_width=True)

    # --- Heatmap of Training Load ---
    elif selected_block == "heatmap_training_load":
    #with st.expander("Heatmap of Training Load"):
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
        st.pyplot(fig8)

    # --- Training Feedback ---
    elif selected_block == "training_feedback":
    #with st.expander("Automatic Training Feedback"):
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
    
    # --- Training Calendar ---
    elif selected_block == "training_calendar":
    #with st.expander("🗓️ Mini Training Calendar by Month"):
        selected_year = st.selectbox("Select Year", sorted(runs['year'].unique(), reverse=True))
        runs_year = runs[runs['year'] == selected_year]

        monthly_summary = runs_year.groupby(['month', 'month_num']).agg({
            'duration_hr': 'sum'
        }).reset_index().sort_values('month_num')

        fig_mini = make_subplots(rows=2, cols=6, subplot_titles=monthly_summary['month'].tolist())

        for i, row in monthly_summary.iterrows():
            m = row['month_num']
            title = row['month']
            dur = int(row['duration_hr'])
            row_num = 1 if i < 6 else 2
            col_num = (i % 6) + 1

            data = runs_year[runs_year['month_num'] == m]
            weekly = data.groupby('week_num')['duration_hr'].sum().reset_index()

            max_y = weekly['duration_hr'].max()
            y_max = max(max_y + 1, 2)

            fig_mini.add_trace(go.Bar(
                x=weekly['week_num'],
                y=weekly['duration_hr'],
                marker_color='black',
                showlegend=False
            ), row=row_num, col=col_num)

            # Add annotation slightly above highest bar (in data coords)
            fig_mini.add_annotation(
                text=f"{dur} HRS",
                x=weekly['week_num'].min() if not weekly.empty else 0,
                y=y_max * 0.90,
                xanchor='left',
                yanchor='bottom',
                showarrow=False,
                row=row_num, col=col_num,
                font=dict(size=12, color='gray')
            )

            fig_mini.update_yaxes(row=row_num, col=col_num, range=[0, y_max])

        fig_mini.update_layout(
            height=600, width=1100,
            title="📅 Monthly Training Calendar — Duration Overview",
            template='plotly_white',
            margin=dict(t=80)
        )

        fig_mini.update_xaxes(showticklabels=False)
        fig_mini.update_yaxes(showticklabels=False)
        st.plotly_chart(fig_mini, use_container_width=True)

    # --- Interactive Calendar View ---
    elif selected_block == "interactive_calendar_view":
    #with st.expander("🗓️ Interactive Calendar View"):
        current_year = datetime.today().year
        current_month = datetime.today().strftime('%b').upper()
        color_map = {
            'Run': 'green', 'Ride': 'blue', 'Walk': 'gray',
            'Workout': 'orange', 'Yoga': 'purple', 'Hike': 'brown'
        }
        default_color = 'lightgray'

        # Filters (side-by-side)
        col1, col2 = st.columns(2)
        with col1:
            years = sorted(runs['year'].unique())
            #selected_year = st.selectbox("Select Year", years)
            selected_year = st.selectbox("Select Year", years, index=years.index(current_year) if current_year in years else 0)
        with col2:
            months = runs[runs['year'] == selected_year]['month'].unique()
            sorted_months = sorted(months, key=lambda x: pd.to_datetime(x, format='%b').month)
            default_month_index = sorted_months.index(current_month) if current_month in sorted_months else 0
            selected_month = st.selectbox("Select Month", sorted_months, index=default_month_index)
            #selected_month = st.selectbox("Select Month", sorted(months, key=lambda x: pd.to_datetime(x, format='%b').month))

        filtered = runs[(runs['year'] == selected_year) & (runs['month'] == selected_month)].copy()
        filtered['date_only'] = filtered['start_date'].dt.date  # Normalize timestamp to date

        daily = (
            filtered.groupby('date_only')
            .agg({
                'distance_km': 'sum',
                'name': lambda x: ' | '.join(x),
                'type': lambda x: ', '.join(sorted(set(x)))
            }).reset_index()
        )

        daily['start_date'] = pd.to_datetime(daily['date_only'])  # Convert back to datetime for further ops
        daily['dow'] = daily['start_date'].dt.day_name()
        daily['week_start'] = daily['start_date'] - pd.to_timedelta(daily['start_date'].dt.weekday, unit='d')

        weekly_totals = (
            daily.groupby('week_start')['distance_km']
            .sum().reset_index()
        )
        weekly_totals['week_label'] = weekly_totals.apply(
            lambda r: f"{r['week_start'].strftime('%b %d')} — {r['distance_km']:.1f} km", axis=1
        )

        # Merge week labels to daily data
        daily = daily.merge(weekly_totals[['week_start', 'week_label']], on='week_start', how='left')

        fig = go.Figure()

        for _, row in daily.iterrows():
            activity_types = row['type'].split(', ')
            main_type = activity_types[0] if len(activity_types) == 1 else 'Mixed'
            color = color_map.get(main_type, default_color)

            fig.add_trace(go.Scatter(
                x=[row['dow']],
                y=[row['week_label']],
                mode='markers+text',
                marker=dict(
                    size=max(10, row['distance_km'] * 5),
                    color=color,
                    opacity=0.8,
                    line=dict(width=1, color='black')
                ),
                text=f"<b>{row['distance_km']:.1f} km</b><br><br><sub>{row['name']}</sub>",
                textposition="middle center",
                textfont=dict(color='black'),
                hovertext=f"{row['name']} ({row['type']})",
                hoverinfo="text"
            ))

        fig.update_layout(
            title=f"Weekly Run Calendar — {selected_month} {selected_year}",
            yaxis=dict(title="Week + Total Distance", autorange="reversed"),
            xaxis=dict(title="Day of Week", categoryorder="array",
                    categoryarray=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
            height=500, width=700, showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)


else:
    st.info("Please upload your exported Strava activity CSV to begin.")
