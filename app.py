import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Load model and encoders
model = joblib.load('ipl_model.pkl')
le_team = joblib.load('le_team.pkl')
le_venue = joblib.load('le_venue.pkl')
le_toss = joblib.load('le_toss.pkl')
le_decision = joblib.load('le_decision.pkl')

# Page config
st.set_page_config(page_title="IPL Winner Predictor", page_icon="🏏", layout="centered")

# Title
st.markdown("<h1 style='text-align:center; color:#F26522;'>🏏 IPL Match Winner Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Predict which team will win based on match conditions</p>", unsafe_allow_html=True)
st.divider()

# Teams and venues list
teams = sorted(le_team.classes_)
venues = sorted(le_venue.classes_)

# Input form
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("🏠 Home Team", teams)
with col2:
    away_team = st.selectbox("✈️ Away Team", teams)

col3, col4 = st.columns(2)
with col3:
    toss_won = st.selectbox("🪙 Toss Winner", [home_team, away_team])
with col4:
    decision = st.selectbox("🏏 Toss Decision", sorted(le_decision.classes_))

venue = st.selectbox("🏟️ Venue", venues)

st.divider()

# Predict button
if st.button("🔮 Predict Winner", use_container_width=True):
    if home_team == away_team:
        st.error("Please select two different teams!")
    else:
        # Encode inputs
        home_enc = le_team.transform([home_team])[0]
        away_enc = le_team.transform([away_team])[0]
        toss_enc = le_toss.transform([toss_won])[0]
        dec_enc = le_decision.transform([decision])[0]
        venue_enc = le_venue.transform([venue])[0]
        home_adv = 1 if toss_won == home_team else 0
        season_num = 2024

        features = np.array([[home_enc, away_enc, toss_enc, dec_enc, venue_enc, home_adv, season_num]])

        # Predict
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

        winner = le_team.inverse_transform([pred])[0]

        # Get probabilities for both teams
        home_idx = list(model.classes_).index(le_team.transform([home_team])[0])
        away_idx = list(model.classes_).index(le_team.transform([away_team])[0])
        home_prob = round(proba[home_idx] * 100, 1)
        away_prob = round(proba[away_idx] * 100, 1)

        # Show result
        st.success(f"🏆 Predicted Winner: **{winner}**")

        # Bar chart
        fig = go.Figure(go.Bar(
            x=[home_team, away_team],
            y=[home_prob, away_prob],
            marker_color=['#F26522', '#1a73e8'],
            text=[f'{home_prob}%', f'{away_prob}%'],
            textposition='auto'
        ))
        fig.update_layout(
            title="Win Probability",
            yaxis_title="Probability %",
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig, use_container_width=True)