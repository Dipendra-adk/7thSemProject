# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler

# # Load dataset
# file_path = "house/Housing.csv"
# df = pd.read_csv(file_path)

# # Check missing values
# print("\nMissing values in each column:")
# print(df.isnull().sum())

# print("\nPercentage of missing values in each column:")
# print((df.isnull().sum() / len(df) * 100).round(2))

# # Handle missing values (additional check for other representations of NaN)
# df.replace({"N/A": np.nan, "": np.nan}, inplace=True)

# # Describe the numerical columns
# print("\nNumerical columns statistics:")
# print(df.describe())

# # Handle missing values
# numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
# categorical_columns = df.select_dtypes(include=['object']).columns

# # For numeric columns, fill with median
# for col in numeric_columns:
#     if df[col].isnull().any():
#         median_value = df[col].median()
#         df[col] = df[col].fillna(median_value)
#         print(f"\nFilled missing values in {col} with median: {median_value}")

# # For categorical columns, fill with mode
# for col in categorical_columns:
#     if df[col].isnull().any():
#         mode_value = df[col].mode()[0]
#         df[col] = df[col].fillna(mode_value)
#         print(f"\nFilled missing values in {col} with mode: {mode_value}")

# # Encode Yes/No categorical variables
# binary_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
# for col in binary_cols:
#     if df[col].isin(["yes", "no"]).all():
#         df[col] = df[col].map({"yes": 1, "no": 0})
#     else:
#         print(f"Warning: '{col}' contains values other than 'yes'/'no'")

# # One-Hot Encoding for 'furnishingstatus'
# if "furnishingstatus" in df.columns:
#     df = pd.get_dummies(df, columns=["furnishingstatus"], dtype=int)
# else:
#     print("Warning: 'furnishingstatus' column not found.")

# # Verify no missing values remain
# print("\nRemaining missing values after cleaning:")
# print(df.isnull().sum())

# # Check the first few rows
# print("\nFirst few rows after cleaning:")
# print(df.head())

# # Save cleaned dataset
# cleaned_file_path = "house/cleaned_house.csv"
# df.to_csv(cleaned_file_path, index=False)
# print(f"\n Data cleaning complete. Cleaned dataset saved at {cleaned_file_path}")



import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Constants for conversion
SQFT_TO_AANA = 342.25  # 1 Aana = 342.25 sq ft
USD_TO_NPR = 80  # 1 USD = 80 NPR

# Load dataset
file_path = "house/Housing.csv"
df = pd.read_csv(file_path)

# Print original units before conversion
print("\nBefore conversion:")
print(f"Area range (sq ft): {df['area'].min():,.2f} to {df['area'].max():,.2f}")
print(f"Price range (USD): ${df['price'].min():,.2f} to ${df['price'].max():,.2f}")

# Convert Area from sq ft to Aana
df['area'] = df['area'] / SQFT_TO_AANA
print("\nConverted area from sq ft to Aana")

# Convert Price from USD to NPR
df['price'] = df['price'] * USD_TO_NPR
print("\nConverted price from USD to NPR")

# Print new units after conversion
print("\nAfter conversion:")
print(f"Area range (Aana): {df['area'].min():,.2f} to {df['area'].max():,.2f}")
print(f"Price range (NPR): Rs.{df['price'].min():,.2f} to Rs.{df['price'].max():,.2f}")

# Check missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nPercentage of missing values in each column:")
print((df.isnull().sum() / len(df) * 100).round(2))

# Handle missing values (additional check for other representations of NaN)
df.replace({"N/A": np.nan, "": np.nan}, inplace=True)

# Describe the numerical columns
print("\nNumerical columns statistics:")
print(df.describe())

# Handle missing values
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

# For numeric columns, fill with median
for col in numeric_columns:
    if df[col].isnull().any():
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
        print(f"\nFilled missing values in {col} with median: {median_value}")

# For categorical columns, fill with mode
for col in categorical_columns:
    if df[col].isnull().any():
        mode_value = df[col].mode()[0]
        df[col] = df[col].fillna(mode_value)
        print(f"\nFilled missing values in {col} with mode: {mode_value}")

# Encode Yes/No categorical variables
binary_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
for col in binary_cols:
    if df[col].isin(["yes", "no"]).all():
        df[col] = df[col].map({"yes": 1, "no": 0})
    else:
        print(f"Warning: '{col}' contains values other than 'yes'/'no'")

# One-Hot Encoding for 'furnishingstatus'
if "furnishingstatus" in df.columns:
    df = pd.get_dummies(df, columns=["furnishingstatus"], dtype=int)
else:
    print("Warning: 'furnishingstatus' column not found.")

# Verify no missing values remain
print("\nRemaining missing values after cleaning:")
print(df.isnull().sum())

# Check the first few rows
print("\nFirst few rows after cleaning:")
print(df.head())

# Add column descriptions
df.columns = df.columns.str.lower()  # Ensure consistent column naming
print("\nColumn descriptions:")
for col in df.columns:
    if col == 'area':
        print(f"{col}: Area in Aana")
    elif col == 'price':
        print(f"{col}: Price in NPR")
    else:
        print(f"{col}: Original values")

# Save cleaned dataset with unit information in filename
cleaned_file_path = "house/cleaned_house.csv"
df.to_csv(cleaned_file_path, index=False)
print(f"\nData cleaning complete. Cleaned dataset saved at {cleaned_file_path}")

# Print sample statistics after conversion
print("\nSummary statistics after conversion:")
print("\nArea (Aana):")
print(f"Mean: {df['area'].mean():,.2f}")
print(f"Median: {df['area'].median():,.2f}")
print(f"Min: {df['area'].min():,.2f}")
print(f"Max: {df['area'].max():,.2f}")

print("\nPrice (NPR):")
print(f"Mean: Rs.{df['price'].mean():,.2f}")
print(f"Median: Rs.{df['price'].median():,.2f}")
print(f"Min: Rs.{df['price'].min():,.2f}")
print(f"Max: Rs.{df['price'].max():,.2f}")
