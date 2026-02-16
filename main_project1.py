import os
from project1.train import train_model_project1
from project1.predict import predict_player_value
from project1.utils import load_and_prepare_data, preprocess_data
from joblib import load
import pandas as pd


MODEL_PATH = 'models/random_forest.joblib'
REAL_VALUES_PATH = 'data/value_players_2020-21.xlsx'
real_values_df = pd.read_excel(REAL_VALUES_PATH)

# ✅ Normaliza los nombres de los jugadores
real_values_df['Player'] = real_values_df['Player'].str.strip().str.lower()

# Corregimos: primero quitamos el símbolo €, luego quitamos la letra "m", y ahora tratamos coma como separador decimal
real_values_df['Market value'] = (
    real_values_df['Market value']
    .str.replace('€', '', regex=False)
    .str.replace('m', '', regex=False)
    .str.replace(',', '', regex=False)  # ← quita separador de miles
)

# Convertimos a número (ya está en millones), luego multiplicamos por 1 millón
real_values_df['Market value'] = pd.to_numeric(real_values_df['Market value'], errors='coerce') * 1e6





def predict_flow():
    from joblib import load

    df = load_and_prepare_data()
    X, _ = preprocess_data(df)

    df_with_info = df.copy()
    df_features = X.copy()

    # Buscar jugador
    player_name = input("\nEnter the name of the player: ").strip().lower()
    matches = df_with_info[df_with_info['player'].str.lower().str.contains(player_name)]

    if matches.empty:
        print("❌ Player not found.")
        return

    # Obtener nombres únicos de jugadores coincidentes
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

    # Filtrar todas las filas de ese jugador exacto
    player_data = df_with_info[df_with_info['player'] == selected_name]
    print("\n📊 Historical market values:")
    for _, row in player_data.iterrows():
        year = row['Season'] if 'Season' in row else row['year']
        value = row['value']
        print(f"- {year}: €{value:,.0f}")

    # Seleccionamos la entrada más reciente
    selected_player = player_data.sort_values('Season', ascending=False).iloc[0]
    player_index = selected_player.name  # índice real en df_features
    input_data = df_features.loc[player_index].to_dict()

    # Cargar modelo y predecir
    model = load(MODEL_PATH)
    predicted_value = predict_player_value(input_data, list(df_features.columns))

    print(f"\nPredicted current market value for {selected_player['player']} for 2020-21: €{predicted_value:,.2f}")
    # Buscar el valor real del jugador para 2020-21
    player_cleaned = selected_player['player'].strip().lower()
    real_row = real_values_df[real_values_df['Player'] == player_cleaned]

    if not real_row.empty:
        real_value = real_row.iloc[0]['Market value']
        print(f"Real market value for 2020-21: €{real_value:,.0f}")
    else:
        print("⚠️ Real market value for 2020-21 not found.")




def menu():
    while True:
        print("\n===== PLAYER VALUE ESTIMATOR =====")
        print("1. Train model")
        print("2. Predict player value")
        print("3. Exit")

        if os.path.exists(MODEL_PATH):
            print("\n✅ Model already trained and available.")
            print("   You can go ahead and predict player values.")
            print("   🔄 Retrain only if you have changed the data or the model code.\n")
        else:
            print("\n⚠️ No trained model found. Please train the model before predicting.\n")

        choice = input("Select an option (1-3): ")

        if choice == '1':
            print("\nTraining model...")
            train_model_project1()
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
