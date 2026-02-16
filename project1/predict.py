import pandas as pd
from joblib import load

def predict_player_value(input_data: dict, columns_used):
    model = load('models/random_forest.joblib')
    df_input = pd.DataFrame([input_data])
    df_input = pd.get_dummies(df_input)

    # Asegurarse de que tiene las mismas columnas que el modelo entrenado
    for col in columns_used:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[columns_used]  # Ordenar columnas igual

    prediction = model.predict(df_input)[0]
    return prediction
