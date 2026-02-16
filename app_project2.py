from project1.eda import (
    show_age_distribution,
    show_avg_value_by_position,
    show_value_by_league,
    show_players_by_foot,
    show_value_distribution_by_foot,
    show_athletes_per_nation,
    show_pairplot_selected,
)

import streamlit as st
from project2.utils import load_all_data, preprocess_data
from project2.train import train_model_project2
from project2.predict import predict_player_value

from joblib import load
import os
import pandas as pd

MODEL_PATH = "models/best_model_project2.joblib"
REAL_VALUES_PATH = 'data/value_players_2020-21.xlsx'

# Load data and real values
st.set_page_config(page_title="Player Market Value Estimator - Project 2", layout="wide")
st.title("⚽ Project 2: Player Market Value Estimator")

df = load_all_data()
real_values_df = pd.read_excel(REAL_VALUES_PATH)
real_values_df['Player'] = real_values_df['Player'].str.strip().str.lower()
real_values_df['Market value'] = (
    real_values_df['Market value']
    .str.replace('€', '', regex=False)
    .str.replace('m', '', regex=False)
    .str.replace(',', '', regex=False)
)
real_values_df['Market value'] = pd.to_numeric(real_values_df['Market value'], errors='coerce') * 1e6

page = st.sidebar.radio("Navigation", ["About", "EDA","Train Model", "Predict Player Value"], key="main_nav2")

# === TRAINING TAB ===
if page == "Train Model":
    st.header("🔁 Train or Retrain Model")
    if st.button("Train Model Now"):
        with st.spinner("Training models. This may take a moment..."):
            best_model_name, rmse = train_model_project2()
        st.success(f"✅ Training complete. Best model: {best_model_name} with RMSE: {rmse:,.2f}")

# === PREDICTION TAB ===
elif page == "Predict Player Value":
    st.header("🔮 Predict Player's Market Value")

    if not os.path.exists(MODEL_PATH):
        st.warning("⚠️ No trained model found. Please train the model first in the 'Train Model' section.")
    else:
        df_raw = load_all_data()
        X, _ = preprocess_data(df_raw)
        df_with_info = df_raw.copy()
        df_features = X.copy()

        player_name = st.text_input("Enter player name:")

        if player_name:
            matches = df_with_info[df_with_info['player'].fillna('').str.lower().str.contains(player_name.lower())]
            unique_players = matches['player'].dropna().unique()

            if len(unique_players) == 0:
                st.error("❌ Player not found.")
            elif len(unique_players) == 1:
                selected_name = unique_players[0]
            else:
                selected_name = st.selectbox("Multiple players found. Please select:", unique_players)

            if selected_name:
                player_rows = matches[matches['player'] == selected_name]
                st.subheader("📊 Historical Market Values:")
                for _, row in player_rows.iterrows():
                    season = row['Season'] if 'Season' in row else "Unknown"
                    value = row['value']
                    st.write(f"- {season}#: €{int(value):,}")

                try:
                    selected_player = player_rows.sort_values('Season', ascending=False).iloc[0]
                    player_index = selected_player.name
                    input_data = df_features.loc[[player_index]]

                    model = load(MODEL_PATH)
                    prediction = predict_player_value(input_data.to_dict(orient='records')[0], list(df_features.columns))
                    st.success(f"🧠 Predicted current market value for {selected_name}: €{prediction:,.2f}")

                    name_lower = selected_name.strip().lower()
                    match = real_values_df[real_values_df['Player'] == name_lower]
                    if not match.empty:
                        real_val = match['Market value'].values[0]
                        st.info(f"📌 Real market value for 2020-21: €{real_val:,.0f}")
                    else:
                        st.warning("⚠️ Real market value for 2020-21 not found.")

                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")

# === ABOUT TAB ===
elif page == "About":
    st.header("ℹ️ About this App")
    st.markdown("""
This is **Project 2** of Álvaro Salvador’s Master's Thesis.

**Goal:** Predict the market value of professional football players using alternative ML algorithms:
- ✅ CatBoost
- ✅ K-Nearest Neighbors (KNN)
- ✅ GradientBoostingRegressor

The best performing model is selected automatically based on RMSE.

**Features:**
- Player name search & intelligent match selection
- Predicted value (based on 2017–2019 data)
- Real 2020–21 market value comparison
- Model retraining available from interface

**Author:** Álvaro Salvador
    """)

    
 # === EDA TAB ===
elif page == "EDA":
    st.header("📊 Exploratory Data Analysis")

    if df is not None:
        st.subheader("1. Age Distribution")
        show_age_distribution(df)   

        st.subheader("2. Average Value by Position")
        show_avg_value_by_position(df)

        st.subheader("3. Value by League")
        show_value_by_league(df)

        st.subheader("4. Value by Foot")
        show_players_by_foot(df)
        show_value_distribution_by_foot(df)

        st.subheader("5. Top Nationalities")
        show_athletes_per_nation(df)

        st.subheader("6. Pairplot (selected variables)")
        pairplot_cols = ['games', 'games_starts', 'minutes', 'goals', 'assists', 'xg', 'npxg', 'xa', 'minutes_90s', 'shots_total']
        show_pairplot_selected(df, pairplot_cols)

    else:
        st.error("❌ Error loading data.")

