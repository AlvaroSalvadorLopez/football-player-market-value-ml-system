import pandas as pd
from joblib import load

MODEL_PATH = 'models/catboost_model_project2.joblib'

def predict_player_value(input_data: dict, columns_used):
    model = load(MODEL_PATH)
    df_input = pd.DataFrame([input_data])
    df_input = pd.get_dummies(df_input)

    for col in columns_used:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[columns_used]

    prediction = model.predict(df_input)[0]
    return prediction
