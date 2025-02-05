
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
     
df=pd.read_csv('house_data.csv')
df.head(5)
df.info()

df1=df.drop(["Title","Year","Views","Build Area","Posted","Amenities"],axis=1)
df1.head(5)
df1.City.value_counts()
city_counts = df.City.value_counts()
values_to_drop= city_counts[city_counts < 80].index.tolist()
print(values_to_drop)
df2= df1.loc[~df1['City'].isin(values_to_drop)]
df2.City.value_counts()
column=(df2.columns.tolist())
print(column)

for col in column:
  num_unique = df2[col].nunique()
  print(f"Number of unique values in 'column_name': {num_unique}")
  print(f"Unique value and count of {col} in Dataframe")
  print(df2[col].value_counts())
  print("\n")

df2.columns.tolist() 
df2.drop(["Address","Bedroom","Bathroom","Parking","Road"],axis=1,inplace=True)
df2.head(10)



# Function to convert each entry to Aana
def convert_to_aana(area):
    area = area.strip()

    # Check for "Ropani" and convert
    if "Ropani" in area:
        ropani = float(re.findall(r'\d+', area)[0])
        return ropani * 16

    # Check for "Sq. Feet" and convert to aana
    if "Sq. Feet" in area:
        # Extract the numeric part using regex
        matches = re.findall(r'\d+', area)
        if matches:  # Ensure there is at least one match
            sq_feet = float(matches[0])
            return sq_feet / 342.25  # Convert to aana

    # Check for a-b-c-d Aana format
    if re.match(r'\d+-\d+-\d+-\d+ Aana', area):
        parts = list(map(int, area.split(' ')[0].split('-')))
        a, b, c, d = parts
        return a * 16 + b + c / 4 + d / 16

    # Check for x-y-z Aana format
    if re.match(r'\d+-\d+-\d+ Aana', area):
        parts = list(map(int, area.split(' ')[0].split('-')))
        x, y, z = parts
        return x + y / 4 + z / 16

    # Check for x-y Aana format
    if re.match(r'\d+-\d+ Aana', area):
        parts = list(map(int, area.split(' ')[0].split('-')))
        x, y = parts
        return x + y / 4

    # Check for "X Aana" format (including decimals)
    if re.match(r'^\d+\.?\d* Aana$', area):
        return float(re.findall(r'\d+\.?\d*', area)[0])

    # If none of the formats match, return "Remove"
    return "Remove"
     

df2['Area'] = df2['Area'].apply(convert_to_aana)
     
df2.head(5)
df2.Area.value_counts().get('Remove',0)
df2 = df2[df2.Area != 'Remove']

df2.shape
nan_count = df2.isna().sum()

print(nan_count)



def convert_to_feet(value):
    if 'Meter' in value:
        # Extract the numerical part, convert to float, and multiply by 3.28084 to convert to feet
        meters = int(float(value.replace(' Meter', '')))
        return meters * 3.28084
    elif 'Feet' in value:
        # Extract the numerical part and return as is
        return int(float(value.replace(' Feet', '')))
    else:
        return "Remove"
     

df3=df2.copy()
     

df3['Road Width'] = df3['Road Width'].apply(convert_to_feet)
     

df3[50:100]
     

df3['Road Width'].value_counts().get('Remove',0)
df3.to_csv('Kathmandu_House_ Price_Prediction_CleanedData.csv', index=False)    
df4 = pd.read_csv('Kathmandu_House_ Price_Prediction_CleanedData.csv')
df4.head(10)
df4["Road Type"].unique()

df4=df3.loc[~df3['Road Type'].isin([' Alley'])]
df4.loc[df4['Road Type'].isin([' Paved', ' Concrete']), 'Road Type'] = 'Blacktopped'
df4['Road Type'] = df4['Road Type'].fillna(' Soil Stabilized')
df4['Road Type'] = df4['Road Type'].str.strip()

df4['Road Type'].value_counts()
df4['Floors'].mean()

df4['Floors'] = df4['Floors'].fillna(2.75)
df4.isna().sum()

df4.to_csv('Kathmandu_House_ Price_Prediction_CleanedData1.csv', index=False)
df4 = pd.read_csv('Kathmandu_House_ Price_Prediction_CleanedData1.csv')
df4 = df4[(df4['Price'] >=10000000 ) & (df4['Price'] <=100000000 )]

plt.figure(figsize=(8, 6))
df4['Price'].plot(kind='line')

# Add labels and title
plt.title('Line Plot of Price')
plt.xlabel('Index')
plt.ylabel('Price in Rupees')

# Show the plot
plt.show()
df4.reset_index(drop=True, inplace=True)
df4.shape
print(df4.index)
df4.head()   

df4 = df4[~((df4['Price'] > 20000000) & (df4['Floors'] < 2) & (df4['Area'] < 4))]
df4 = df4[~((df4['Price'] > 20000000) & (df4['Area'] < 2.5))]
df4 = df4[~((df4['Price'] > 50000000) & (df4['Floors'] < 3) & (df4['Area'] < 5))]
df4 = df4[~((df4['Price'] < 50000000) & (df4['Area'] > 15))]
df4 = df4[~((df4['Price'] < 16000000) & (df4['Floors'] > 2.5))]

df4.shape
df4.head(50)

df_perAana = pd.DataFrame({'Price_per_aana': df4['Price'] / df4['Area']})
df_perAana.head(5)

df_perAana.replace([float('inf'), -float('inf')], np.nan, inplace=True)
df_perAana.dropna(inplace=True)

# Check for NaN values
nan_count = df_perAana['Price_per_aana'].isna().sum()
print(f"Number of NaN values: {nan_count}")

# Check for infinite values
inf_count = (df_perAana['Price_per_aana'] == float('inf')).sum()
print(f"Number of infinite values: {inf_count}")

plt.figure(figsize=(8, 6))
bins = [3000000, 4000000, 5000000, 6000000, 7000000, 8000000]
plt.hist(df_perAana['Price_per_aana'], bins = bins,rwidth=0.8)
plt.xlim(3000000, 8000000)
plt.title('Histogram of Price per Aana')
plt.xlabel('Price per Aana')
plt.ylabel('Frequency')
plt.show()

df4.to_csv('Kathmandu_House_ Price_Prediction_CleanedData2.csv', index=False)
df4.shape
df4 = pd.read_csv('Kathmandu_House_ Price_Prediction_CleanedData2.csv')
df4 = df4[~((df4['Area'] < 2.0))]
df4.columns = df4.columns.str.replace(' ', '_')
df4.columns
df4.head(5)

avg_price_per_city = df4.groupby('City')['Price'].mean()
avg_price_per_city.plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Average Price per City')
plt.xlabel('City')
plt.ylabel('Average Price')

avg_price_per_city = df4.groupby('Face')['Price'].mean()
avg_price_per_city.plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Average Price per City')
plt.xlabel('Face')
plt.ylabel('Average Price')

avg_price_per_city = df4.groupby('Road_Type')['Price'].mean()
avg_price_per_city.plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Average Price per City')
plt.xlabel('Road Type')
plt.ylabel('Average Price')

print(df4.dtypes)
df4.drop('Face',axis=1,inplace=True)

# Apply one-hot encoding to only 'City' and 'Road Type'
encoded_columns = pd.get_dummies(df4[['City', 'Road_Type']], dtype=int)

# Drop the original 'City' and 'Road Type' columns
df4 = df4.drop(['City', 'Road_Type'], axis=1)

# Concatenate the one-hot encoded columns to the DataFrame
df4 = pd.concat([df4, encoded_columns], axis=1)

df4.head()
print(df4.dtypes)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df4['Price'] = scaler.fit_transform(df4[['Price']])
df4.head()

df4.to_csv('Final_cleaned_encoded_data.csv', index=False)