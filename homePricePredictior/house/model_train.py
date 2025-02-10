# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error
# from ml_models.svm_model import SVR
# from ml_models.decision_tree import DecisionTreeRegressor

# # Load dataset
# data = pd.read_csv("house/Final_cleaned_encoded_data.csv")

# # Handle missing values
# data.fillna(data.median(), inplace=True)

# # Separate features and target variable
# X = data.drop("Price", axis=1)
# y = data["Price"]

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# # Create separate scalers for features and target
# feature_scaler = StandardScaler()
# target_scaler = StandardScaler()

# # Scale features
# X_train_scaled = feature_scaler.fit_transform(X_train)
# X_test_scaled = feature_scaler.transform(X_test)

# # Scale target variable
# y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
# y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

# ### Train Decision Tree Model
# tree_regressor = DecisionTreeRegressor(max_depth=5, min_samples_split=10)
# tree_regressor.fit(X_train_scaled, y_train_scaled)

# # Predictions
# y_pred_tree_scaled = tree_regressor.predict(X_test_scaled)

# # Inverse transform predictions to original scale
# y_pred_tree = target_scaler.inverse_transform(y_pred_tree_scaled.reshape(-1, 1)).ravel()

# # Evaluate Decision Tree
# r2_tree = r2_score(y_test, y_pred_tree)
# rmse_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))
# print("Decision Tree Model Performance:")
# print(f"R² Score: {r2_tree:.4f}")
# print(f"RMSE: {rmse_tree:.4f}\n")
# # Save feature names
# feature_names = X.columns.tolist()
# with open('house/ml_models/feature_names.pkl', 'wb') as f:
#     pickle.dump(feature_names, f)
    
# # Save Decision Tree Model and Scalers
# pickle.dump(tree_regressor, open("house/ml_models/saved_models/decision_tree.pkl", "wb"))
# pickle.dump(feature_scaler, open("house/ml_models/saved_models/feature_scaler.pkl", "wb"))
# pickle.dump(target_scaler, open("house/ml_models/saved_models/target_scaler.pkl", "wb"))

# ### Train SVM Model
# svr_regressor = SVR(C=5.0, epsilon=0.05, learning_rate=0.01, n_epochs=2000)
# svr_regressor.fit(X_train_scaled, y_train_scaled)

# # Predictions
# y_pred_svm_scaled = svr_regressor.predict(X_test_scaled)

# # Inverse transform predictions to original scale
# y_pred_svm = target_scaler.inverse_transform(y_pred_svm_scaled.reshape(-1, 1)).ravel()

# # Evaluate SVM Model
# r2_svm = r2_score(y_test, y_pred_svm)
# rmse_svm = np.sqrt(mean_squared_error(y_test, y_pred_svm))
# print("SVM Model Performance:")
# print(f"R² Score: {r2_svm:.4f}")
# print(f"RMSE: {rmse_svm:.4f}\n")

# # Save SVM Model
# pickle.dump(svr_regressor, open("house/ml_models/saved_models/svm_model.pkl", "wb"))

# print("Models trained and saved successfully!")

# model_train.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import pickle

# Load dataset
data = pd.read_csv("house/Final_cleaned_encoded_data.csv")

# Print initial data info
print("Initial data shape:", data.shape)
print("\nInitial price statistics:")
print(data["Price"].describe())

# Remove any rows with zero or negative prices
data = data[data["Price"] > 0]
print("\nData shape after removing invalid prices:", data.shape)

# Handle missing values
data = data.fillna(data.median())

# Remove any remaining rows with NaN values
data = data.dropna()
print("\nFinal data shape:", data.shape)

# Verify data cleanup
print("\nFinal price statistics:")
print(data["Price"].describe())

# Separate features and target variable
X = data.drop("Price", axis=1)
y = data["Price"]

# Apply log transformation to price (target variable) after ensuring all values are positive
y_log = np.log1p(y)

print("\nLog-transformed price statistics:")
print(y_log.describe())

# Verify no NaN values in transformed data
print("\nNumber of NaN values in features:", X.isnull().sum().sum())
print("Number of NaN values in target:", y_log.isnull().sum())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.15, random_state=42)

# Save train and test datasets
# Combine features and target for saving
train_data = pd.concat([pd.DataFrame(X_train, columns=X.columns), 
                       pd.Series(y_train, name='Price_Log')], axis=1)
test_data = pd.concat([pd.DataFrame(X_test, columns=X.columns), 
                      pd.Series(y_test, name='Price_Log')], axis=1)

# Add original price values
train_data['Price_Original'] = np.expm1(train_data['Price_Log'])
test_data['Price_Original'] = np.expm1(test_data['Price_Log'])

# Save to CSV
train_data.to_csv('media/datasets/train_dataset.csv', index=False)
test_data.to_csv('media/datasets/test_dataset.csv', index=False)

print("\nDatasets saved:")
print(f"Training set shape: {train_data.shape}")
print(f"Test set shape: {test_data.shape}")


# Scale features
feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

# Train Decision Tree Model
tree_regressor = DecisionTreeRegressor(
    max_depth=100,
    min_samples_split=5,
    min_samples_leaf=4,
    random_state=42
)
tree_regressor.fit(X_train_scaled, y_train)

# Train SVM Model
svr_regressor = SVR(
    kernel='rbf',
    C=500.0,
    epsilon=0.01,
    gamma='scale'
)
svr_regressor.fit(X_train_scaled, y_train)

# Evaluate models
def evaluate_model(model, X_scaled, y_true, model_name):
    y_pred_log = model.predict(X_scaled)
    y_pred = np.expm1(y_pred_log)
    y_true_original = np.expm1(y_true)
    
    r2 = r2_score(y_true_original, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true_original, y_pred))
    
    print(f"\n{model_name} Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    return r2, rmse

# Evaluate both models
dt_r2, dt_rmse = evaluate_model(tree_regressor, X_test_scaled, y_test, "Decision Tree")
svm_r2, svm_rmse = evaluate_model(svr_regressor, X_test_scaled, y_test, "SVM")

# Save feature names
feature_names = X.columns.tolist()
print("\nFeatures being saved:", feature_names)

# Save models and scaler
pickle.dump(tree_regressor, open("house/ml_models/saved_models/decision_tree.pkl", "wb"))
pickle.dump(svr_regressor, open("house/ml_models/saved_models/svm_model.pkl", "wb"))
pickle.dump(feature_scaler, open("house/ml_models/saved_models/feature_scaler.pkl", "wb"))
pickle.dump(feature_names, open("house/ml_models/feature_names.pkl", "wb"))

print("\nModels trained and saved successfully!")

# # model_train.py
# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error
# from ml_models.svm_model import SVR
# from ml_models.decision_tree import DecisionTreeRegressor

# # Load dataset
# data = pd.read_csv("house/Final_cleaned_encoded_data.csv")

# # Handle missing values
# data.fillna(data.median(), inplace=True)

# # Remove any negative or zero prices
# data = data[data['Price'] > 0]

# # Separate features and target variable
# X = data.drop("Price", axis=1)
# y = data["Price"]

# # Print some statistics about the price data
# print("Price Statistics Before Transform:")
# print(y.describe())
# print("\nMin price:", y.min())
# print("Max price:", y.max())
# print("Number of zeros or negative values:", (y <= 0).sum())

# # Log transform the target variable
# y = np.log1p(y)

# # Verify the transformation worked
# print("\nPrice Statistics After Transform:")
# print(y.describe())
# print("Number of NaN values:", y.isna().sum())

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# # Scale features
# feature_scaler = StandardScaler()
# X_train_scaled = feature_scaler.fit_transform(X_train)
# X_test_scaled = feature_scaler.transform(X_test)

# print("\nTraining Decision Tree Model...")
# # Train Decision Tree Model
# tree_regressor = DecisionTreeRegressor(max_depth=5, min_samples_split=10)
# tree_regressor.fit(X_train_scaled, y_train)

# # Predictions
# y_pred_tree = tree_regressor.predict(X_test_scaled)

# # Transform back predictions
# y_pred_tree_orig = np.expm1(y_pred_tree)
# y_test_orig = np.expm1(y_test)

# # Check for NaN values
# print("\nChecking for NaN values:")
# print("NaN in predictions:", np.isnan(y_pred_tree_orig).sum())
# print("NaN in test set:", np.isnan(y_test_orig).sum())

# # Remove any NaN values before calculating metrics
# mask = ~np.isnan(y_pred_tree_orig) & ~np.isnan(y_test_orig)
# y_pred_tree_orig_clean = y_pred_tree_orig[mask]
# y_test_orig_clean = y_test_orig[mask]

# # Evaluate Decision Tree
# r2_tree = r2_score(y_test_orig_clean, y_pred_tree_orig_clean)
# rmse_tree = np.sqrt(mean_squared_error(y_test_orig_clean, y_pred_tree_orig_clean))
# print("\nDecision Tree Model Performance:")
# print(f"R² Score: {r2_tree:.4f}")
# print(f"RMSE: {rmse_tree:.4f}")

# # Save feature names
# feature_names = X.columns.tolist()
# with open('house/ml_models/feature_names.pkl', 'wb') as f:
#     pickle.dump(feature_names, f)

# # Save Decision Tree Model and Scaler
# pickle.dump(tree_regressor, open("house/ml_models/saved_models/decision_tree.pkl", "wb"))
# pickle.dump(feature_scaler, open("house/ml_models/saved_models/feature_scaler.pkl", "wb"))

# print("\nTraining SVM Model...")
# # Train SVM Model
# svr_regressor = SVR(C=5.0, epsilon=0.05, learning_rate=0.01, n_epochs=2000)
# svr_regressor.fit(X_train_scaled, y_train)

# # Predictions
# y_pred_svm = svr_regressor.predict(X_test_scaled)

# # Transform back predictions
# y_pred_svm_orig = np.expm1(y_pred_svm)

# # Remove any NaN values before calculating metrics
# mask = ~np.isnan(y_pred_svm_orig) & ~np.isnan(y_test_orig)
# y_pred_svm_orig_clean = y_pred_svm_orig[mask]
# y_test_orig_clean = y_test_orig[mask]

# # Evaluate SVM
# r2_svm = r2_score(y_test_orig_clean, y_pred_svm_orig_clean)
# rmse_svm = np.sqrt(mean_squared_error(y_test_orig_clean, y_pred_svm_orig_clean))
# print("\nSVM Model Performance:")
# print(f"R² Score: {r2_svm:.4f}")
# print(f"RMSE: {rmse_svm:.4f}")

# # Save SVM Model
# pickle.dump(svr_regressor, open("house/ml_models/saved_models/svm_model.pkl", "wb"))

# print("\nModels trained and saved successfully!")

# # Save the minimum and maximum prices for later use
# price_info = {
#     'min_price': float(data['Price'].min()),
#     'max_price': float(data['Price'].max()),
#     'mean_price': float(data['Price'].mean())
# }
# with open('house/ml_models/price_info.pkl', 'wb') as f:
#     pickle.dump(price_info, f)


