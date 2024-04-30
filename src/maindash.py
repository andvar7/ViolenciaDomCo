import dash
import dash_bootstrap_components as dbc
import pandas as pd
import json
import joblib
from urllib.request import urlopen

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.MATERIA, dbc.icons.FONT_AWESOME],
)
app.title = "ViolenciaDoméstica"
server = app.server

# import the dataset
url = "data_cleaned.parquet"
df = pd.read_parquet(url)

with urlopen('https://gist.githubusercontent.com/john-guerra/43c7656821069d00dcbc/raw/be6a6e239cd5b5b803c6e7c2ec405b793a9064dd/Colombia.geo.json') as response:
    counties = json.load(response)

X_test = "Xtest_cleaned.csv"
X_test = pd.read_csv(X_test)

y_test = "ytest_cleaned.csv"
y_test = pd.read_csv(y_test)

reg = pd.read_parquet('data_cluster.parquet')


models = {
    "Support Vector Machine": joblib.load("models/SVM_model.joblib"),
    "Multi-layer Perceptron": joblib.load("models/MLP_model.joblib"),
    "Random Forest": joblib.load("models/Random_Forest_model.joblib"),
    "XGBoost": joblib.load("models/XGBoost_model.joblib"),
}