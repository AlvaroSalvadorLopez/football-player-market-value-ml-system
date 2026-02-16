import os
from joblib import load
from project2.train import train_model_project2

from project2.predict import predict_player_value
from project1.utils import load_and_prepare_data, preprocess_data
import pandas as pd

MODEL_PATH = 'models/best_model_project2.joblib'

REAL_VALUES_PATH = 'data/value_players_2020-21.xlsx'

real_values_df = pd.read_excel(REAL_VALUES_PATH)
real_values_df['Player'] = real_values_df['Player'].str.strip().str.lower()
real_values_df['Market value'] = (
    real_values_df['Market value']
    .str.replace('€', '', regex=False)
    .str.replace('m', '', regex=False)
    .str.replace(',', '', regex=False)
)
real_values_df['Market value'] = pd.to_numeric(real_values_df['Market value'], errors='coerce') * 1e6

def predict_flow():
    df = load_and_prepare_data()
    X, _ = preprocess_data(df)
    df_with_info = df.copy()
    df_features = X.copy()

    player_name = input("\nEnter the name of the player: ").strip().lower()
    matches = df_with_info[df_with_info['player'].str.lower().str.contains(player_name)]

    if matches.empty:
        print("❌ Player not found.")
        return

    unique_players = matches['player'].unique()

    if len(unique_players) > 1:
        print("\nMultiple matches found:")
        for i, name in enumerate(unique_players, 1):
            print(f"[{i}] {name}")
        try:
            selected_index = int(input("Enter the number of the player you want to select: ")) - 1
            selected_name = unique_players[selected_index]
        except (ValueError, IndexError):
            print("❌ Invalid selection.")
            return
    else:
        selected_name = unique_players[0]

    player_data = df_with_info[df_with_info['player'] == selected_name]
    print("\n📊 Historical market values:")
    for _, row in player_data.iterrows():
        year = row['Season'] if 'Season' in row else row['year']
        value = row['value']
        print(f"- {year}: €{value:,.0f}")

    selected_player = player_data.sort_values('Season', ascending=False).iloc[0]
    player_index = selected_player.name
    input_data = df_features.loc[player_index].to_dict()

    model = load(MODEL_PATH)
    predicted_value = predict_player_value(input_data, list(df_features.columns))
    print(f"\n🧠 Predicted current market value for {selected_name}: €{predicted_value:,.2f}")

    name_lower = selected_name.strip().lower()
    match = real_values_df[real_values_df['Player'] == name_lower]
    if not match.empty:
        real_val = match['Market value'].values[0]
        print(f"📌 Real market value for 2020-21: €{real_val:,.0f}")
    else:
        print("⚠️ Real market value for 2020-21 not found.")

def menu():
    while True:
        print("\n===== PLAYER VALUE ESTIMATOR – PROJECT 2 (CatBoost) =====")
        print("1. Train model")
        print("2. Predict player value")
        print("3. Exit")

        if os.path.exists(MODEL_PATH):
            print("\n✅ Model already trained and available.")
        else:
            print("\n⚠️ No trained model found. Please train it first.")

        choice = input("Select an option (1-3): ")

        if choice == '1':
            print("\nTraining model...")
            train_model_project2()

            print("✅ Model trained and saved.")
        elif choice == '2':
            if not os.path.exists(MODEL_PATH):
                print("⚠️ Model not trained yet. Please train it first.")
            else:
                predict_flow()
        elif choice == '3':
            print("Exiting program.")
            break
        else:
            print("Invalid option. Please choose 1, 2 or 3.")

if __name__ == "__main__":
    menu()
