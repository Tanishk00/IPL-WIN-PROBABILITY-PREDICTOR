import streamlit as st
import pandas as pd
import joblib
import os

# --- Static Options ---
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# --- Load Model ---
MODEL_PATH = 'pipe.pkl'

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: `{MODEL_PATH}`")
        return None
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

pipe = load_model()
if not pipe:
    st.stop()

# --- UI ---
st.title('🏏 IPL Win Predictor')

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('🏏 Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('🎯 Select the bowling team', sorted(teams))

selected_city = st.selectbox('🏙️ Select host city', sorted(cities))

target = st.number_input('🎯 Target Runs', min_value=1, step=1)

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('🏏 Current Score', min_value=0, step=1)
with col4:
    overs = st.number_input('⏱️ Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets_out = st.number_input('❌ Wickets Out', min_value=0, max_value=10, step=1)

# --- Predict Button ---
if st.button('🔮 Predict Probability'):
    runs_left = target - score
    balls_left = 120 - int(overs * 6)
    wickets = 10 - wickets_out

    crr = 0 if overs == 0 else round(score / overs, 2)
    rrr = 0 if balls_left == 0 else round((runs_left * 6) / balls_left, 2)

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Predict probability
    try:
        result = pipe.predict_proba(input_df)
        loss_prob = result[0][0]
        win_prob = result[0][1]

        st.subheader("📊 Win Prediction")
        st.success(f"{batting_team} Win Probability: **{round(win_prob * 100)}%**")
        st.error(f"{bowling_team} Win Probability: **{round(loss_prob * 100)}%**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
