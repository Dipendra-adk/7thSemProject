import pandas as pd
from decision_tree import DecisionTreeRegressor

csv_path = r"D:\web development project\7thSemProject\homePricePredictior\house\Final_cleaned_encoded_data.csv"
df = pd.read_csv(csv_path)

features = ['Floors', 'Area', 'Road_Width', 'City_Bhaktapur', 'City_Kathmandu', 'City_Lalitpur', 'Road_Type_Blacktopped', 'Road_Type_Gravelled', 'Road_Type_Soil Stabilized']
X = df[features]
y = df['Price']

dt_reg = DecisionTreeRegressor() 
dt_reg.fit(X, y)

feature_importance = dt_reg.feature_importances_

for feature, importance in zip(features, feature_importance):
    print(f"{feature}: {importance:.4f}")

print(df.describe(include="all"))
