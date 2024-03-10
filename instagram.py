import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


data = pd.read_csv('instagram_reach.csv', encoding='utf-8')
data.drop(['Unnamed: 0', 'S.No'], axis=1, inplace=True)
data['Caption'].fillna('', inplace=True)
data['Time since posted'] = data['Time since posted'].str.extract('(\d+)').astype(int)

X = data[['Followers', 'Time since posted']]
y = data['Likes']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r_squared = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'RMSE: {rmse}')
print(f'R-Squared: {r_squared}')


