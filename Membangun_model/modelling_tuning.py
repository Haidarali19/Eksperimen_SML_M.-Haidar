import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Inisialisasi DagsHub
dagshub.init(repo_owner='Haidarali19', repo_name='Eksperimen_SML_M.-Haidar', mlflow=True)

# Load Data
df = pd.read_csv('house_prices_preprocessing.csv')
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow Experiment dengan Hyperparameter Tuning
with mlflow.start_run(run_name="HousePrice_RF_Tuning"):
    
    # parameter untuk tuning
    n_estimators = 100
    max_depth = 15
    
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Prediksi & Metriks
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Logging ke MLflow
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    
    # Artefak 1: Plot Actual vs Predicted
    plt.figure(figsize=(10,6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Harga Asli")
    plt.ylabel("Prediksi Harga")
    plt.title("Actual vs Predicted House Prices")
    plt.savefig("actual_vs_predicted.png")
    mlflow.log_artifact("actual_vs_predicted.png")

    # Artefak 2: Feature Importance Plot
    plt.figure(figsize=(10,6))
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")

    # Log Model ke MLflow
    mlflow.sklearn.log_model(model, "house_price_model")

    print(f"Eksperimen selesai! MAE: {mae}, R2: {r2}")