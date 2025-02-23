import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import pickle

data = pd.read_csv("house/Final_cleaned_encoded_data.csv")


print("Initial data shape:", data.shape)
print("\nInitial price statistics:")
print(data["Price"].describe())


data = data[data["Price"] > 0]
print("\nData shape after removing invalid prices:", data.shape)


data = data.fillna(data.median())


data = data.dropna()
print("\nFinal data shape:", data.shape)


print("\nFinal price statistics:")
print(data["Price"].describe())


X = data.drop("Price", axis=1)
y = data["Price"]


y_log = np.log1p(y)

print("\nLog-transformed price statistics:")
print(y_log.describe())

print("\nNumber of NaN values in features:", X.isnull().sum().sum())
print("Number of NaN values in target:", y_log.isnull().sum())


X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.15, random_state=42)


train_data = pd.concat([pd.DataFrame(X_train, columns=X.columns), 
                       pd.Series(y_train, name='Price_Log')], axis=1)
test_data = pd.concat([pd.DataFrame(X_test, columns=X.columns), 
                      pd.Series(y_test, name='Price_Log')], axis=1)

train_data['Price_Original'] = np.expm1(train_data['Price_Log'])
test_data['Price_Original'] = np.expm1(test_data['Price_Log'])


train_data.to_csv('media/datasets/train_dataset.csv', index=False)
test_data.to_csv('media/datasets/test_dataset.csv', index=False)

print("\nDatasets saved:")
print(f"Training set shape: {train_data.shape}")
print(f"Test set shape: {test_data.shape}")


feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)


tree_regressor = DecisionTreeRegressor(
    max_depth=100,
    min_samples_split=5,
    min_samples_leaf=4,
    random_state=42
)
tree_regressor.fit(X_train_scaled, y_train)


svr_regressor = SVR(
    kernel='rbf',
    C=500.0,
    epsilon=0.01,
    gamma='scale'
)
svr_regressor.fit(X_train_scaled, y_train)


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


dt_r2, dt_rmse = evaluate_model(tree_regressor, X_test_scaled, y_test, "Decision Tree")
svm_r2, svm_rmse = evaluate_model(svr_regressor, X_test_scaled, y_test, "SVM")


feature_names = X.columns.tolist()
print("\nFeatures being saved:", feature_names)


pickle.dump(tree_regressor, open("house/ml_models/saved_models/decision_tree.pkl", "wb"))
pickle.dump(svr_regressor, open("house/ml_models/saved_models/svm_model.pkl", "wb"))
pickle.dump(feature_scaler, open("house/ml_models/saved_models/feature_scaler.pkl", "wb"))
pickle.dump(feature_names, open("house/ml_models/feature_names.pkl", "wb"))

print("\nModels trained and saved successfully!")
