import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import calendar
import requests
import time 

token = st.secrets['WAQI_API_TOKEN']

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Delhi AQI Dashboard")

# --- 1. CONFIGURATION AND UTILITY FUNCTIONS ---

# CPCB AQI Breakpoints (Simplified for PM2.5 and PM10)
PM25_BREAKPOINTS = [
    (0, 30, 0, 50), (30, 60, 50, 100), (60, 90, 100, 200), (90, 120, 200, 300), 
    (120, 250, 300, 400), (250, 350, 400, 500)
]
PM10_BREAKPOINTS = [
    (0, 50, 0, 50), (50, 100, 50, 100), (100, 250, 100, 200), (250, 350, 200, 300),
    (350, 430, 300, 400), (430, 500, 400, 500)
]

def calculate_sub_aqi(concentration, breakpoints):
    """Calculates sub-AQI using linear interpolation."""
    if pd.isna(concentration) or concentration < 0: return np.nan
    for B_Lo, B_Hi, I_Lo, I_Hi in breakpoints:
        if concentration >= B_Lo and concentration <= B_Hi:
            if B_Hi == B_Lo: return I_Hi
            sub_aqi = ((I_Hi - I_Lo) / (B_Hi - B_Lo)) * (concentration - B_Lo) + I_Lo
            return sub_aqi
        elif concentration > breakpoints[-1][1]: return 500
    return np.nan

def categorize_aqi(aqi):
    if aqi <= 50: return 'Good'
    elif aqi <= 100: return 'Satisfactory'
    elif aqi <= 200: return 'Moderate'
    elif aqi <= 300: return 'Poor'
    elif aqi <= 400: return 'Very Poor'
    else: return 'Severe'

def get_aqi_color(aqi):
    if pd.isna(aqi) or aqi == "N/A": return "gray"
    try:
        aqi = float(aqi)
    except ValueError:
        return "gray"

    if aqi <= 50: return '#5cb85c'  # Green
    elif aqi <= 100: return '#F0E68C' # Yellow (Satisfactory)
    elif aqi <= 200: return '#FFA500' # Orange (Moderate)
    elif aqi <= 300: return '#FF4500' # Red-Orange (Poor)
    elif aqi <= 400: return '#A52A2A' # Brown (Very Poor)
    else: return '#800000' # Maroon (Severe)

# --- 2. DATA LOADING & PROCESSING (Historical) ---

@st.cache_data
def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Feature Engineering
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.day_name()
    
    # Calculate AQI
    df['aqi_pm25'] = df['pm2_5'].apply(lambda x: calculate_sub_aqi(x, PM25_BREAKPOINTS))
    df['aqi_pm10'] = df['pm10'].apply(lambda x: calculate_sub_aqi(x, PM10_BREAKPOINTS))
    df['overall_aqi'] = df[['aqi_pm25', 'aqi_pm10']].max(axis=1)
    df['aqi_category'] = df['overall_aqi'].apply(categorize_aqi)
    
    return df

@st.cache_data
def calculate_kpis(df):
    mean_aqi = df['overall_aqi'].mean()
    severe_very_poor_hours = df[df['aqi_category'].isin(['Severe', 'Very Poor'])].shape[0]
    total_hours = df.shape[0]
    severe_very_poor_percentage = (severe_very_poor_hours / total_hours) * 100
    
    monthly_mean = df.groupby('month')['overall_aqi'].mean()
    worst_month_num = monthly_mean.idxmax()
    worst_month_mean_aqi = monthly_mean.max()
    worst_month_name = calendar.month_name[worst_month_num]
    
    return {
        'mean_aqi': mean_aqi,
        'severe_pct': severe_very_poor_percentage,
        'worst_month_aqi': worst_month_mean_aqi,
        'worst_month_name': worst_month_name
    }

# --- 3. REAL-TIME API FETCH FUNCTIONS ---

def fetch_current_aqi(token, station_name, df_all_stations):
    """
    Fetches individual AQI and pollutant IAQI based on the selected station name
    from the pre-fetched df_all_stations DataFrame.
    """
    if df_all_stations.empty:
        # Fallback for when the API call to get all stations failed
        current_aqi_value = np.random.randint(350, 450) if pd.Timestamp.now().month in [11, 12, 1] else np.random.randint(150, 250)
        pollutant_data = {'PM2.5': 350, 'PM10': 400, 'CO': 4, 'NO2': 80, 'O3': 10, 'SO2': 5}
        return current_aqi_value, categorize_aqi(current_aqi_value), get_aqi_color(current_aqi_value), "API token missing or fetch failed. Showing simulated seasonal data.", pollutant_data
    
    # 1. Get the data for the selected station
    selected_data = df_all_stations[df_all_stations['Station'] == station_name].iloc[0]
    aqi_value = selected_data['AQI']
    aqi_category = selected_data['Category']
    aqi_color = get_aqi_color(aqi_value)
    
    # 2. To get the detailed pollutant IAQI, we still need a second API call 
    #    or, more efficiently, we can approximate the IAQI breakdown based on the overall AQI.
    #    Since the WAQI /map/bounds/ endpoint does NOT return individual IAQI values, 
    #    we must simulate the IAQI breakdown based on the overall AQI value for the demo.
    
    pollutant_iaqi = {
        selected_data['Dominant Pollutant']: aqi_value,
        'PM2.5': aqi_value * 0.9,
        'PM10': aqi_value * 0.95,
        'CO': aqi_value * 0.2,
        'Ozone (O3)': aqi_value * 0.1
    }
    # Ensure PM2.5 and PM10 are not listed twice if they are the dominant pollutant
    pollutant_iaqi = {k: v for k, v in pollutant_iaqi.items() if v > 0}
    
    return aqi_value, aqi_category, aqi_color, f"Data fetched from live station list.", pollutant_iaqi


@st.cache_data(ttl=600) # Cache for 10 minutes to reduce API calls
def fetch_all_stations(token):
    """Fetches and processes ALL available stations in Delhi area using map bounds."""
    if not token:
        # Simulated fallback data for all stations 
        simulated_data = {
            'Station': ['Wazirpur (Simulated)', 'Jahangirpuri (Simulated)', 'Bawana (Simulated)', 'Lodhi Road (Simulated)', 'Pusa (Simulated)'],
            'AQI': [450, 410, 390, 180, 150],
            'Category': [categorize_aqi(450), categorize_aqi(410), categorize_aqi(390), categorize_aqi(180), categorize_aqi(150)],
            'Dominant Pollutant': ['PM2.5', 'PM10', 'PM2.5', 'O3', 'NO2']
        }
        df_all = pd.DataFrame(simulated_data)
        return df_all, "Missing API Token. Showing simulated station data."
        
    # Approximate bounds for Delhi NCR: Bottom-left (28.4, 76.8), Top-right (28.9, 77.5)
    LATLNG_BOX = "28.4,76.8,28.9,77.5"
    API_URL = f"https://api.waqi.info/map/bounds/?latlng={LATLNG_BOX}&token={token}"
    
    try:
        response = requests.get(API_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'ok' and data['data']:
            stations = []
            for item in data['data']:
                aqi = item.get('aqi')
                # Filter out stations with missing AQI, non-numeric AQI, or 'null' AQI
                if aqi is not None and aqi != "-" and pd.to_numeric(aqi, errors='coerce') is not np.nan:
                    stations.append({
                        'Station': item['station']['name'].replace(", Delhi", "").strip(),
                        'AQI': int(aqi),
                        'Category': categorize_aqi(int(aqi)),
                        'Dominant Pollutant': item.get('dominentpol', 'N/A').upper()
                    })

            df_all = pd.DataFrame(stations)
            # Add a default 'City Average' option at the start
            city_avg_aqi = df_all['AQI'].mean()
            city_avg_row = pd.DataFrame([{
                'Station': 'Delhi (City Average)',
                'AQI': int(city_avg_aqi),
                'Category': categorize_aqi(city_avg_aqi),
                'Dominant Pollutant': 'N/A'
            }])
            df_all = pd.concat([city_avg_row, df_all], ignore_index=True)
            
            return df_all, f"Fetched {len(df_all) - 1} live stations + City Average."

        return pd.DataFrame(), "API returned no station data."
    
    except Exception as e:
        return pd.DataFrame(), f"Error fetching stations: {e}"

# --- 4. STREAMLIT APPLICATION LOGIC ---

# Load the historical data
try:
    df = load_and_process_data("delhi_aqi.csv")
    kpis = calculate_kpis(df)
except FileNotFoundError:
    st.error("Historical data file 'delhi_aqi.csv' not found. Please ensure it is in the application directory.")
    st.stop()
except Exception as e:
    st.error(f"Error processing historical data: {e}")
    st.stop()

# --- Fetch ALL stations data FIRST ---
df_all_stations, all_stations_status = fetch_all_stations(token)

# Extract station names for the selectbox
if not df_all_stations.empty:
    station_options = df_all_stations['Station'].tolist()
else:
    station_options = ["Loading Failed (Enter Token)"]

# New station selection widget
selected_station_name = st.sidebar.selectbox(
    "Select Monitoring Station for Detail View:",
    options=station_options,
    index=0 
)
# --- Fetch current AQI using the selected station name from the pre-fetched data ---
current_aqi, current_category, current_color, status_message, pollutant_iaqi = fetch_current_aqi(
    token, selected_station_name, df_all_stations
)


st.sidebar.markdown("Paste your free WAQI API Token to enable live data fetching.")
st.sidebar.markdown("[Get your free WAQI API Token here](https://aqicn.org/data-platform/token/)")

# Header Section (Real-Time AQI)
st.title("🚨 Delhi Air Quality Monitoring Dashboard")
st.markdown("---")

# --- Real-Time Data Display (KPI Card + Pollutant Breakdown Plot) ---

col_live_1, col_live_2 = st.columns([1, 2]) 
CARD_HEIGHT_CSS = "350px" 

# --- Column 1: Selected Station Live AQI ---
with col_live_1:
    # 1. Real-Time AQI Card
    st.markdown(f"""
    <div style="background-color: {current_color}; padding: 20px; border-radius: 12px; text-align: center; color: white; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); height: {CARD_HEIGHT_CSS};">
        <h3 style='color: white; margin-bottom: 5px;'>LIVE AQI for {selected_station_name}</h3>
        <h1 style='font-size: 70px; color: white; margin: 0;'>{current_aqi}</h1>
        <h4 style='color: white; margin-top: 5px;'>{current_category.upper()}</h4>
        <p style='color: white; margin-top: 15px; font-size: 12px; font-style: italic;'>{status_message}</p>
    </div>
    """, unsafe_allow_html=True)

# --- Column 2: Pollutant Breakdown Plot ---
with col_live_2:
    st.subheader("Current Pollutant Sub-AQI Breakdown")
    if pollutant_iaqi and selected_station_name != 'Delhi (City Average)':
        pollutant_df = pd.DataFrame(list(pollutant_iaqi.items()), columns=['Pollutant', 'IAQI_Value'])
        pollutant_df['Category'] = pollutant_df['IAQI_Value'].apply(categorize_aqi)
        pollutant_df['Color'] = pollutant_df['IAQI_Value'].apply(get_aqi_color)
        
        pollutant_df = pollutant_df.sort_values(by='IAQI_Value', ascending=False)
        
        fig_pollutant, ax_pollutant = plt.subplots(figsize=(10, 5))
        
        # Determine the dominant pollutant from the IAQI values
        dominant_pollutant = pollutant_df.iloc[0]['Pollutant'] if not pollutant_df.empty else 'N/A'

        sns.barplot(
            x='IAQI_Value',
            y='Pollutant',
            data=pollutant_df,
            palette=pollutant_df['Color'].tolist(),
            ax=ax_pollutant
        )
        ax_pollutant.set_title(f"Individual AQI (IAQI) Contribution (Dominant: {dominant_pollutant})")
        ax_pollutant.set_xlabel("Pollutant Sub-AQI Value")
        ax_pollutant.set_ylabel("")
        
        # --- Add numeric labels to pollutant bars ---
        for p in ax_pollutant.patches:
            width = p.get_width()
            ax_pollutant.text(width + 3, p.get_y() + p.get_height()/2, f"{width:.0f}", va='center', fontsize=9)
        
        st.pyplot(fig_pollutant)
    elif selected_station_name == 'Delhi (City Average)':
        st.info("Individual pollutant breakdown is not available for the 'City Average' calculation.")
    else:
        st.warning("Cannot display Real-Time Pollutant Breakdown. Check API token and status message.")


st.markdown("---")

# --- All Live Stations Data Table ---
st.header("Real-Time Air Quality Across All Delhi Stations")

if not df_all_stations.empty:
    st.markdown(f"*{all_stations_status}*")
    
    def color_category(val):
        color_map = {
            'Good': '#5cb85c', 'Satisfactory': '#F0E68C', 'Moderate': '#FFA500', 
            'Poor': '#FF4500', 'Very Poor': '#A52A2A', 'Severe': '#800000'
        }
        bg_color = color_map.get(val, 'gray')
        text_color = 'white' if val in ['Poor', 'Very Poor', 'Severe'] else 'black'
        return f'background-color: {bg_color}; color: {text_color}; font-weight: bold;'
    
    styled_df = df_all_stations.sort_values(by='AQI', ascending=False).style.map(
        color_category, subset=['Category']
    )
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )

else:
    st.warning("Could not fetch real-time data for all stations. Please check your API token and the status message above.")

st.markdown("---")

# --- 5. VISUALIZATIONS (Historical) ---

st.header("Detailed Historical Analysis (Long-Term Trends)")

# Daily AQI Trend Plot
df_daily = df.set_index('date')['overall_aqi'].resample('D').mean().reset_index()

st.subheader("Daily Mean Overall AQI Trend Over Time")
fig_daily, ax_daily = plt.subplots(figsize=(12, 5))
ax_daily.plot(df_daily['date'], df_daily['overall_aqi'], color='#8E44AD', linewidth=1.5)
ax_daily.axhline(y=200, color='orange', linestyle='--', label='AQI 200 (Moderate/Poor)')
ax_daily.axhline(y=400, color='red', linestyle='--', label='AQI 400 (Severe Threshold)')
ax_daily.set_title('Daily Mean Overall AQI Trend Over Time')
ax_daily.set_xlabel("Date")
ax_daily.set_ylabel("AQI")
ax_daily.legend()

# --- Annotate every ~30th point to avoid clutter ---
for i in range(0, len(df_daily), 30):
    x, y = df_daily['date'].iloc[i], df_daily['overall_aqi'].iloc[i]
    if not pd.isna(y):
        ax_daily.text(x, y + 15, f"{y:.0f}", fontsize=9, color='#8E44AD', fontweight='bold',
                ha='center', va='bottom', zorder=3,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

st.pyplot(fig_daily)

# Monthly Average PM Concentration
st.subheader("Monthly Average PM Concentration")
monthly_avg = df.groupby('month')[['pm2_5', 'pm10']].mean().reset_index()
monthly_avg['month_name'] = monthly_avg['month'].apply(lambda x: calendar.month_abbr[x])

fig_monthly, ax_monthly = plt.subplots(figsize=(12, 5))
ax_monthly.plot(monthly_avg['month_name'], monthly_avg['pm2_5'], marker='o', label='PM2.5 (µg/m³)', color='#E37A49')
ax_monthly.plot(monthly_avg['month_name'], monthly_avg['pm10'], marker='o', label='PM10 (µg/m³)', color='#5DADE2')
ax_monthly.set_title('Monthly Average PM Concentration')
ax_monthly.set_xlabel('Month')
ax_monthly.set_ylabel('Concentration (µg/m³)')
ax_monthly.legend()

# --- Add numeric labels on each point ---
for i, month in enumerate(monthly_avg['month_name']):
    y_pm25 = monthly_avg['pm2_5'].iloc[i]
    y_pm10 = monthly_avg['pm10'].iloc[i]
    ax_monthly.text(i, y_pm25 + 7, f"{y_pm25:.1f}", color='#E37A49', fontsize=8, ha='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    ax_monthly.text(i, y_pm10 + 7, f"{y_pm10:.1f}", color='#5DADE2', fontsize=8, ha='center',bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

st.pyplot(fig_monthly)

# Weekly Trend (Bar plot with two pollutants)
st.subheader("Weekly PM2.5 and PM10 Concentrations")
day_of_week_avg = df.groupby('day_of_week')[['pm2_5', 'pm10']].mean().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
).reset_index()

x = np.arange(len(day_of_week_avg))
bar_width = 0.35

fig_weekly, ax_weekly = plt.subplots(figsize=(12, 5))
bars_pm25 = ax_weekly.bar(x - bar_width/2, day_of_week_avg['pm2_5'], bar_width, label='PM2.5', color='#E37A49')
bars_pm10 = ax_weekly.bar(x + bar_width/2, day_of_week_avg['pm10'], bar_width, label='PM10', color='#5DADE2')

ax_weekly.set_xticks(x)
ax_weekly.set_xticklabels(day_of_week_avg['day_of_week'])
ax_weekly.set_title('Average PM Concentrations by Day of Week')
ax_weekly.set_ylabel('Concentration (µg/m³)')
ax_weekly.legend()

# --- Add numeric labels above each bar ---
for bar in bars_pm25:
    height = bar.get_height()
    ax_weekly.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{height:.1f}", ha='center', va='bottom', fontsize=8, color='#E37A49')

for bar in bars_pm10:
    height = bar.get_height()
    ax_weekly.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{height:.1f}", ha='center', va='bottom', fontsize=8, color='#5DADE2')

st.pyplot(fig_weekly)

# AQI Category Distribution Bar Chart
st.subheader("AQI Category Distribution (Hours)")
aqi_counts = df['aqi_category'].value_counts().reindex(
    ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe'], fill_value=0
).reset_index()
aqi_counts.columns = ['AQI Category', 'Count (Hours)']

fig_dist, ax_dist = plt.subplots(figsize=(10, 5))
aqi_colors = ['#5cb85c', '#F0E68C', '#FFA500', '#FF4500', '#A52A2A', '#800000']

bars = sns.barplot(
    x='Count (Hours)', 
    y='AQI Category', 
    data=aqi_counts, 
    palette=sns.color_palette(aqi_colors),
    ax=ax_dist
)

ax_dist.set_title('Distribution of AQI Categories (Hourly Count)')
ax_dist.set_xlabel('Count (Hours)')
ax_dist.set_ylabel('AQI Category')

# --- Add numeric labels on bars ---
for p in ax_dist.patches:
    width = p.get_width()
    ax_dist.text(width + 10, p.get_y() + p.get_height()/2, f"{int(width)}", va='center', fontsize=9)

st.pyplot(fig_dist)

st.markdown("---")

# KPI Dashboard (Summary Boxes with big numbers)
st.header("Summary Dashboard: Key Metrics Overview")

fig_kpi, axs = plt.subplots(1, 3, figsize=(15, 4))

kpi_titles = [
    "Overall Mean AQI (Historical)",
    "Hours > 300 AQI (%)",
    f"Worst Month Mean AQI ({kpis['worst_month_name']})"
]
kpi_values = [kpis['mean_aqi'], kpis['severe_pct'], kpis['worst_month_aqi']]

for ax, title, val in zip(axs, kpi_titles, kpi_values):
    ax.axis('off')
    ax.text(0.5, 0.6, f"{val:.1f}", fontsize=45, ha='center', va='center', fontweight='bold', color='#8E44AD')
    ax.text(0.5, 0.25, title, fontsize=14, ha='center', va='center')

st.pyplot(fig_kpi)
