import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

def prepare_data(ttc_df):
    features = ['Route', 'Min_Gap', 'Report_Year', 'Report_Day']
    X = pd.get_dummies(ttc_df[features], drop_first=True)
    y = ttc_df['Min_Delay']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_save_model():
    
    df = pd.read_csv('TTC_Cleaned.csv')  

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R²: {r2:.3f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Save model and list of columns to keep for prediction
    joblib.dump(model, 'linear_regression_model.pkl')
    joblib.dump(X_train.columns.tolist(), 'model_columns.pkl')

if __name__ == "__main__":
    train_and_save_model()
