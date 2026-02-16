from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from joblib import dump
from math import sqrt

from project1.utils import load_and_prepare_data, preprocess_data

def train_model_project1():
    df = load_and_prepare_data()
    X, y = preprocess_data(df)

    # ⚠️ LightGBM removed – no need to clean feature names
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "RandomForest": RandomForestRegressor(),
        "XGBoost": XGBRegressor(),
        "SVR": SVR()
    }

    best_rmse = float("inf")
    best_model = None
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = sqrt(mean_squared_error(y_test, y_pred))
        print(f"{name} RMSE: {rmse:.2f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_name = name

    # Save best model
    model_path = 'models/best_model_project1.joblib'
    dump(best_model, model_path)
    print(f"\n✅ Best model: {best_name} with RMSE: {best_rmse:.2f}")
    print(f"Model saved to {model_path}")

    return best_name, best_rmse


