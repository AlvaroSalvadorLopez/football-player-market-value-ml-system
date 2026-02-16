import pandas as pd

def load_and_prepare_data():
    df_1718 = pd.read_csv('data/transfermarkt_fbref_201718.csv', delimiter=';', index_col=0, low_memory=False)
    df_1819 = pd.read_csv('data/transfermarkt_fbref_201819.csv', delimiter=';', index_col=0, low_memory=False)
    df_1920 = pd.read_csv('data/transfermarkt_fbref_201920.csv', delimiter=';', index_col=0, low_memory=False)

    df_1718['year'] = 2017
    df_1819['year'] = 2018
    df_1920['year'] = 2019

    df = pd.concat([df_1718, df_1819, df_1920], ignore_index=True)
    df = df.dropna(subset=['value'])
    df.index = range(len(df))
    return df

def preprocess_data(df):
    if 'Attendance' in df.columns:
        df = df.drop(columns=['Attendance'])
    if 'Season' in df.columns:
        df = df.drop(columns=['Season'])

    y = df['value']
    X = df.drop(columns=['value'])

    X = pd.get_dummies(X, drop_first=True)

    X = X.fillna(0)
    y = y[X.index]

    return X, y

def load_all_data():
    df_1718 = pd.read_csv('data/transfermarkt_fbref_201718.csv', delimiter=';', index_col=0, low_memory=False)
    df_1819 = pd.read_csv('data/transfermarkt_fbref_201819.csv', delimiter=';', index_col=0, low_memory=False)
    df_1920 = pd.read_csv('data/transfermarkt_fbref_201920.csv', delimiter=';', index_col=0, low_memory=False)

    # ✅ Añadir columna 'year' para que coincida con los datos de entrenamiento
    df_1718['year'] = 2017
    df_1819['year'] = 2018
    df_1920['year'] = 2019

    df_all = pd.concat([df_1718, df_1819, df_1920], ignore_index=True)
    return df_all


