import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from ml_models.decision_tree import DecisionTreeRegressor
from ml_models.svm_model import SVMRegressor

def engineer_features(df, is_training=False):
    """Create consistent features for both training and prediction"""
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Basic feature engineering
    df_processed['total_rooms'] = df_processed['bedrooms'] + df_processed['bathrooms']
    df_processed['room_density'] = df_processed['total_rooms'] / df_processed['area']
    df_processed['parking_ratio'] = df_processed['parking'] / df_processed['total_rooms']
    
    # Create luxury score
    luxury_features = ['airconditioning', 'hotwaterheating', 'guestroom', 'basement', 'prefarea']
    df_processed['luxury_score'] = df_processed[luxury_features].sum(axis=1)
    
    # Log transform area
    df_processed['area'] = np.log1p(df_processed['area'])
    
    if is_training:
        # Calculate price per area using original area value
        original_area = np.expm1(df_processed['area'])
        df_processed['price_per_area'] = df_processed['price'] / original_area
        # Log transform price for training
        df_processed['price'] = np.log1p(df_processed['price'])
        
        # Store feature names for prediction
        feature_names = [col for col in df_processed.columns if col not in ['price', 'price_per_area']]
        with open("house/ml_models/feature_names.pkl", "wb") as f:
            pickle.dump(feature_names, f)
            
        return df_processed
    
    return df_processed

def preprocess_data():
    """Load and preprocess the training data"""
    df = pd.read_csv("house/cleaned_house.csv")
    
    # Process features
    df_processed = engineer_features(df, is_training=True)
    
    # Prepare features and target
    y = df_processed['price']
    X = df_processed.drop(['price', 'price_per_area'], axis=1)
    
    return X, y

def train_and_evaluate_models():
    """Train and evaluate models, save train/test data, and store trained models"""
    print("Loading and preprocessing data...")
    X, y = preprocess_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # Save train and test data
    os.makedirs("house/dataset", exist_ok=True)
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    train_data.to_csv("house/dataset/train.csv", index=False)
    test_data.to_csv("house/dataset/test.csv", index=False)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter tuning for Decision Tree
    print("\nTuning Decision Tree...")
    dt_param_grid = {
        'max_depth': [150, 200, 300, 500],
        'min_samples_split': [50, 100, 200],
        'min_samples_leaf': [5, 10, 20]
    }
    dt_model = GridSearchCV(DecisionTreeRegressor(), dt_param_grid, cv=5, n_jobs=-1)
    dt_model.fit(X_train, y_train)
    
    # Hyperparameter tuning for SVM
    print("\nTuning SVM...")
    svm_param_grid = {
        'C': [0.1, 1, 10, 50, 100],
        'epsilon': [0.01, 0.05, 0.1, 0.2],
        'learning_rate': [0.0001, 0.001, 0.01],
        'n_epochs': [1000, 2000]
    }
    svm_model = GridSearchCV(SVMRegressor(), svm_param_grid, cv=5, n_jobs=-1)
    svm_model.fit(X_train_scaled, y_train)

    # Make predictions
    dt_preds = dt_model.predict(X_test)
    svm_preds = svm_model.predict(X_test_scaled)
    
    # Transform predictions back to original scale
    y_test_original = np.expm1(y_test)
    dt_preds_original = np.expm1(dt_preds)
    svm_preds_original = np.expm1(svm_preds)
    
    # Calculate metrics
    dt_r2 = r2_score(y_test_original, dt_preds_original)
    dt_rmse = np.sqrt(mean_squared_error(y_test_original, dt_preds_original))
    svm_r2 = r2_score(y_test_original, svm_preds_original)
    svm_rmse = np.sqrt(mean_squared_error(y_test_original, svm_preds_original))
    
    print("\nModel Performance:")
    print(f"Decision Tree: R² = {dt_r2:.4f}, RMSE = {dt_rmse:.2f}")
    print(f"SVM: R² = {svm_r2:.4f}, RMSE = {svm_rmse:.2f}")

    # Save models and scaler
    os.makedirs("house/ml_models/saved_models", exist_ok=True)
    with open("house/ml_models/saved_models/decision_tree.pkl", "wb") as f:
        pickle.dump(dt_model.best_estimator_, f)
    with open("house/ml_models/saved_models/svm_model.pkl", "wb") as f:
        pickle.dump(svm_model.best_estimator_, f)
    with open("house/ml_models/saved_models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return dt_model.best_estimator_, svm_model.best_estimator_, scaler

if __name__ == "__main__":
    try:
        dt_model, svm_model, scaler = train_and_evaluate_models()
        print("\nModels trained and saved successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
