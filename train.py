import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

# Load the dataset
data = pd.read_csv('dataset/day.csv')  # Ensure this is the correct path to your CSV file

# Convert the 'dteday' column to datetime format
# data['dteday'] = pd.to_datetime(data['dteday'], format='%d-%m-%Y')

# Handle missing values (if any)
print(data.isnull().sum())  # Check for missing values

# Convert categorical features to numerical (One-Hot Encoding)
categorical_cols = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Define features and target variable
X = data.drop(['cnt', 'instant', 'dteday', 'casual', 'registered'], axis=1)  # Drop target and unnecessary columns
y = data['cnt']  # 'cnt' is the target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Training set shape: {X_train.shape}, Test set shape: {X_test.shape}')

# Set up MLflow
mlflow.set_tracking_uri("file:./mlruns")  # Make sure this is the directory for logging
experiment_name = "BikeSharing_Experiment"
mlflow.set_experiment(experiment_name)  # Create or set the experiment

# Train and log Random Forest Regressor
with mlflow.start_run(run_name="RandomForest_BikeSharing"):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # Evaluate the model
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)  # RMSE

    print(f'Random Forest MSE: {mse_rf}')
    print(f'Random Forest MAE: {mae_rf}')
    print(f'Random Forest R²: {r2_rf}')
    print(f'Random Forest RMSE: {rmse_rf}')

    # Log metrics and model in MLflow
    mlflow.log_metric("mse", mse_rf)
    mlflow.log_metric("mae", mae_rf)
    mlflow.log_metric("r2", r2_rf)
    mlflow.log_metric("rmse", rmse_rf)
    mlflow.sklearn.log_model(rf_model, "RandomForest_Model")

# Train and log Linear Regression Model
with mlflow.start_run(run_name="LinearRegression_BikeSharing"):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    # Evaluate the model
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)  # RMSE

    print(f'Linear Regression MSE: {mse_lr}')
    print(f'Linear Regression MAE: {mae_lr}')
    print(f'Linear Regression R²: {r2_lr}')
    print(f'Linear Regression RMSE: {rmse_lr}')

    # Log metrics and model in MLflow
    mlflow.log_metric("mse", mse_lr)
    mlflow.log_metric("mae", mae_lr)
    mlflow.log_metric("r2", r2_lr)
    mlflow.log_metric("rmse", rmse_lr)
    mlflow.sklearn.log_model(lr_model, "LinearRegression_Model")
