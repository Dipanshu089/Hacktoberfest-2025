import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Page configuration
st.set_page_config(
    page_title="Smart Traffic Insights - Indore",
    page_icon="🚦",
    layout="wide"
)

# Load and preprocess traffic dataset with fallback
@st.cache_data
def load_traffic_dataset():
    """Load traffic.csv or create dummy data if not found"""
    try:
        # Try to load real dataset
        df = pd.read_csv('traffic.csv')
        st.success("✅ Real traffic.csv loaded!")
        
        # Convert DateTime to proper format
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        
        # Extract time-based features
        df['hour'] = df['DateTime'].dt.hour
        df['dayofweek'] = df['DateTime'].dt.dayofweek
        df['month'] = df['DateTime'].dt.month
        df['year'] = df['DateTime'].dt.year
        
        # One-hot encode Junction column
        junction_dummies = pd.get_dummies(df['Junction'], prefix='Junction')
        df = pd.concat([df, junction_dummies], axis=1)
        
        # Store junction names for UI
        junction_names = sorted(df['Junction'].unique().tolist())
        
        return df, junction_names, True
        
    except FileNotFoundError:
        st.warning("⚠️ traffic.csv not found. Using dummy data for demo.")
        return create_dummy_data()
    except Exception as e:
        st.error(f"❌ Error loading traffic.csv: {str(e)}. Using dummy data.")
        return create_dummy_data()

def create_dummy_data():
    """Create dummy traffic data matching the real dataset structure"""
    np.random.seed(42)
    
    # Create dummy data similar to your real dataset
    dates = pd.date_range(start='2015-11-01', end='2015-11-30', freq='H')
    junctions = [1, 2, 3, 4]  # Common junction IDs
    
    data = []
    for date in dates:
        for junction in junctions:
            hour = date.hour
            day_of_week = date.weekday()
            
            # Simulate realistic traffic patterns
            if hour in [7, 8, 17, 18, 19]:  # Rush hours
                base_vehicles = np.random.normal(25, 8)
            elif hour in [9, 10, 11, 14, 15, 16]:  # Moderate traffic
                base_vehicles = np.random.normal(18, 5)
            elif hour in [12, 13]:  # Lunch time
                base_vehicles = np.random.normal(22, 6)
            else:  # Low traffic
                base_vehicles = np.random.normal(12, 4)
            
            # Weekend adjustment
            if day_of_week >= 5:
                base_vehicles *= 0.7
            
            vehicles = max(1, int(base_vehicles + np.random.normal(0, 2)))
            
            # Create ID similar to real data
            date_str = date.strftime('%Y%m%d%H')
            id_val = f"{date_str}{junction}"
            
            data.append({
                'DateTime': date,
                'Junction': junction,
                'Vehicles': vehicles,
                'ID': id_val,
                'hour': hour,
                'dayofweek': day_of_week,
                'month': date.month,
                'year': date.year
            })
    
    df = pd.DataFrame(data)
    
    # One-hot encode Junction column
    junction_dummies = pd.get_dummies(df['Junction'], prefix='Junction')
    df = pd.concat([df, junction_dummies], axis=1)
    
    junction_names = sorted(df['Junction'].unique().tolist())
    
    return df, junction_names, False

# Get junction columns for model features
@st.cache_data
def get_junction_columns():
    """Get the list of junction columns from the dataset"""
    df, _, _ = load_traffic_dataset()
    if df is not None:
        junction_cols = [col for col in df.columns if col.startswith('Junction_')]
        return junction_cols
    return []

# Load model with fallback
@st.cache_resource
def load_model():
    try:
        model = joblib.load('traffic_model.pkl')
        st.success("✅ Your trained model loaded successfully!")
        return model, True
    except FileNotFoundError:
        st.warning("⚠️ traffic_model.pkl not found. Using dummy model for demo.")
        return create_dummy_model(), False
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}. Using dummy model.")
        return create_dummy_model(), False

def create_dummy_model():
    """Create a dummy model with exactly 5 features to match the real model"""
    df, _, _ = load_traffic_dataset()
    
    # Use the same 5 features as the real model: hour, dayofweek, Junction_2, Junction_3, Junction_4
    # (Junction_1 is dropped as reference category)
    feature_cols = ['hour', 'dayofweek', 'Junction_2', 'Junction_3', 'Junction_4']
    
    # Ensure all junction columns exist
    for col in ['Junction_2', 'Junction_3', 'Junction_4']:
        if col not in df.columns:
            df[col] = 0  # Add missing columns as zeros
    
    X = df[feature_cols]
    y = df['Vehicles']
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model

# Main app
st.title("🚦 Smart Traffic Insights - Indore")

# Load data and model
df, junction_names, is_real_data = load_traffic_dataset()
model, is_real_model = load_model()

# Sidebar controls
st.sidebar.title("🚦 Controls")

# Junction selector (replacing road selector)
selected_junctions = st.sidebar.multiselect(
    "Select Junctions", 
    options=junction_names, 
    default=junction_names[:3] if len(junction_names) >= 3 else junction_names
)

time_range = st.sidebar.slider("Hour Range", 0, 23, (6, 22))

# Filter data
filtered_df = df[
    (df['Junction'].isin(selected_junctions)) &
    (df['hour'] >= time_range[0]) &
    (df['hour'] <= time_range[1])
]

# Key metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    avg_vehicles = filtered_df['Vehicles'].mean() if 'Vehicles' in filtered_df.columns else 0
    st.metric("Avg Vehicles", f"{avg_vehicles:.0f}")
with col2:
    high_traffic = (filtered_df['Vehicles'] > filtered_df['Vehicles'].quantile(0.75)).mean() * 100 if 'Vehicles' in filtered_df.columns else 0
    st.metric("High Traffic %", f"{high_traffic:.1f}%")
with col3:
    st.metric("Junctions Monitored", len(selected_junctions))
with col4:
    st.metric("Data Points", len(filtered_df))

# Model upload section
with st.expander("📁 Upload Your Model"):
    st.code("joblib.dump(your_model, 'traffic_model.pkl')")
    uploaded_model = st.file_uploader("Upload .pkl file", type=['pkl'])
    if uploaded_model:
        with open('traffic_model.pkl', 'wb') as f:
            f.write(uploaded_model.read())
        st.success("Model saved! Restart app to use it.")

# Traffic prediction with real model features
st.subheader("🔮 Traffic Prediction")

def prepare_model_features(hour, dayofweek, selected_junction, junction_names):
    """Prepare features exactly as your model expects (5 features total)"""
    
    # Most likely your model was trained with: hour, dayofweek, and 3 junction dummies (dropping one for multicollinearity)
    # Standard practice is to drop the first category to avoid dummy variable trap
    
    # Create base features
    features_dict = {
        'hour': hour,
        'dayofweek': dayofweek
    }
    
    # One-hot encode junction (drop Junction_1 to avoid multicollinearity - standard ML practice)
    # This gives us exactly 5 features: hour, dayofweek, Junction_2, Junction_3, Junction_4
    for junction in [2, 3, 4]:  # Skip junction 1 (reference category)
        junction_col = f'Junction_{junction}'
        features_dict[junction_col] = 1 if junction == selected_junction else 0
    
    # Feature order for 5 features: hour, dayofweek, Junction_2, Junction_3, Junction_4
    feature_order = ['hour', 'dayofweek', 'Junction_2', 'Junction_3', 'Junction_4']
    
    features_array = np.array([[features_dict.get(col, 0) for col in feature_order]])
    
    # Debug info
    st.write(f"🔍 Debug: Features prepared - {dict(zip(feature_order, features_array[0]))}")
    st.write(f"📊 Feature array shape: {features_array.shape}")
    
    return features_array

col1, col2, col3 = st.columns(3)
with col1:
    pred_hour = st.selectbox("Hour (0-23)", range(24), index=9)
with col2:
    pred_dayofweek = st.selectbox("Day of Week", 
                                 ["Monday (0)", "Tuesday (1)", "Wednesday (2)", "Thursday (3)", 
                                  "Friday (4)", "Saturday (5)", "Sunday (6)"], index=0)
with col3:
    pred_junction = st.selectbox("Junction", junction_names)

if st.button("Predict Traffic"):
    # Extract day number from selection
    dayofweek_num = int(pred_dayofweek.split('(')[1].split(')')[0])
    
    try:
        # Prepare features exactly as your model expects
        features = prepare_model_features(pred_hour, dayofweek_num, pred_junction, junction_names)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Determine congestion level based on your data distribution
        if prediction > filtered_df['Vehicles'].quantile(0.75):
            level, color = "High", "🔴"
        elif prediction > filtered_df['Vehicles'].quantile(0.5):
            level, color = "Medium", "🟡"
        else:
            level, color = "Low", "🟢"
        
        st.success(f"Predicted Vehicles: **{prediction:.0f}**")
        st.info(f"Traffic Level: {color} **{level}**")
        st.info(f"📍 Junction: **{pred_junction}** | 🕐 Hour: **{pred_hour}** | 📅 Day: **{pred_dayofweek.split('(')[0].strip()}**")
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.info("💡 Make sure your model features match: hour, dayofweek, and one-hot encoded junctions")

# Visualizations with real data
st.subheader("📊 Analytics")

if 'Vehicles' in filtered_df.columns:
    # Traffic by hour
    hourly_data = filtered_df.groupby('hour')['Vehicles'].mean().reset_index()
    fig1 = px.line(hourly_data, x='hour', y='Vehicles', 
                   title='Vehicle Count by Hour')
    st.plotly_chart(fig1, use_container_width=True)

    # Traffic by junction
    col1, col2 = st.columns(2)
    with col1:
        junction_data = filtered_df.groupby('Junction')['Vehicles'].mean().sort_values(ascending=True)
        fig2 = px.bar(x=junction_data.values, y=junction_data.index, orientation='h',
                      title='Average Vehicles by Junction')
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # Traffic distribution
        q25, q75 = filtered_df['Vehicles'].quantile([0.25, 0.75])
        bins = pd.cut(filtered_df['Vehicles'], 
                     bins=[0, q25, q75, filtered_df['Vehicles'].max()], 
                     labels=['Low', 'Medium', 'High'])
        dist_data = bins.value_counts()
        fig3 = px.pie(values=dist_data.values, names=dist_data.index,
                      title='Traffic Distribution',
                      color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'})
        st.plotly_chart(fig3, use_container_width=True)

    # Day of week analysis
    if len(filtered_df) > 0:
        dow_data = filtered_df.groupby('dayofweek')['Vehicles'].mean().reset_index()
        dow_data['day_name'] = dow_data['dayofweek'].map({
            0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'
        })
        fig4 = px.bar(dow_data, x='day_name', y='Vehicles',
                      title='Average Vehicles by Day of Week')
        st.plotly_chart(fig4, use_container_width=True)

# Business insights based on real data
st.subheader("💡 Key Insights")
if 'Vehicles' in filtered_df.columns and len(filtered_df) > 0:
    peak_hour = filtered_df.groupby('hour')['Vehicles'].mean().idxmax()
    busiest_junction = filtered_df.groupby('Junction')['Vehicles'].mean().idxmax()
    weekend_avg = filtered_df[filtered_df['dayofweek'].isin([5, 6])]['Vehicles'].mean()
    weekday_avg = filtered_df[~filtered_df['dayofweek'].isin([5, 6])]['Vehicles'].mean()
    
    insights = [
        f"🚦 Peak hour: {peak_hour}:00 shows highest traffic",
        f"📍 Busiest junction: {busiest_junction}",
        f"📅 Weekend vs Weekday: {((weekend_avg/weekday_avg - 1) * 100):+.1f}% difference",
        f"⚡ Traffic range: {filtered_df['Vehicles'].min():.0f} - {filtered_df['Vehicles'].max():.0f} vehicles"
    ]
    
    for insight in insights:
        st.markdown(f"- {insight}")

# File upload section
st.subheader("📁 Upload Files")
col1, col2 = st.columns(2)

with col1:
    st.write("**Dataset Upload**")
    uploaded_dataset = st.file_uploader("Upload traffic.csv", type=['csv'])
    if uploaded_dataset:
        with open('traffic.csv', 'wb') as f:
            f.write(uploaded_dataset.read())
        st.success("Dataset saved! Restart app to use it.")

with col2:
    st.write("**Model Upload**")
    uploaded_model = st.file_uploader("Upload traffic_model.pkl", type=['pkl'])
    if uploaded_model:
        with open('traffic_model.pkl', 'wb') as f:
            f.write(uploaded_model.read())
        st.success("Model saved! Restart app to use it.")

# Footer
st.markdown("---")
st.markdown("**Smart Traffic Insights for Indore** | Built with Real Data & AI")