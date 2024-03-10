import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('Bengaluru_House_Data.csv')
print(df.head())

df_info = df.info()
df_describe = df.describe(include='all')

categorical_columns = ['area_type', 'availability', 'location', 'size', 'society']
unique_values = {col: df[col].nunique() for col in categorical_columns}

print('Missing Values and Data Types:')
print(df_info)
print('\
Summary Statistics:')
print(df_describe)
print('\
Unique Values in Categorical Columns:')
print(unique_values)

df_info = df.info()
df_describe = df.describe(include='all')

# Handling missing values
# For simplicity, we'll fill missing values in 'size' and 'bath' with the mode, and 'balcony' with 0 (assuming no balcony if not specified)
df['size'].fillna(df['size'].mode()[0], inplace=True)
df['bath'].fillna(df['bath'].mode()[0], inplace=True)
df['balcony'].fillna(0, inplace=True)

# For 'location' and 'society', we'll fill missing values with 'Unknown'
df['location'].fillna('Unknown', inplace=True)
df['society'].fillna('Unknown', inplace=True)

# Check if there are any missing values left
missing_values_after = df.isnull().sum()

print('Missing Values After Cleaning:')
print(missing_values_after)

def clean_total_sqft(value):
    try:
        if '-' in value:  # Handle ranges
            low, high = map(float, value.split('-'))
            return (low + high) / 2
        elif 'Sq. Meter' in value:  # Convert from square meters to square feet
            return float(value.split('Sq. Meter')[0]) * 10.7639
        else:  # Handle single numeric values
            return float(value)
    except:
        return np.nan  # Return NaN for values that cannot be handled

# Clean the 'total_sqft' column

df['total_sqft'] = df['total_sqft'].apply(lambda x: clean_total_sqft(x))

# Display the first few rows to verify the changes
print(df.head())

# Check for any NaN values in 'total_sqft'
print('\
NaN values in total_sqft:', df['total_sqft'].isna().sum())

# Set the aesthetic style of the plots
sns.set_style('whitegrid')

# Exploratory Data Analysis (EDA)

# Price Distribution
plt.figure(figsize=(10, 6), facecolor='white')
plt.title('Distribution of House Prices')
sns.histplot(df['price'], kde=True, bins=30)
plt.xlabel('Price (in Lakhs)')
plt.ylabel('Frequency')
plt.show()

# Relationship between Total Square Feet and Price
plt.figure(figsize=(10, 6), facecolor='white')
plt.title('Total Square Feet vs. Price')
plt.scatter(df['total_sqft'], df['price'], alpha=0.5)
plt.xlabel('Total Square Feet')
plt.ylabel('Price (in Lakhs)')
plt.show()

# Correlation Matrix
plt.figure(figsize=(10, 6), facecolor='white')
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

df['price_per_sqft'] = df['price']*100000 / df['total_sqft']

# Categorizing properties based on their size
# Assuming categories based on total square feet: Small (<1000), Medium (1000-2000), Large (>2000)
df['size_category'] = pd.cut(df['total_sqft'], bins=[0, 1000, 2000, float('inf')], labels=['Small', 'Medium', 'Large'], right=False)

# Categorizing properties based on the number of bathrooms
# Assuming categories: 1-2 (Few), 3-4 (Moderate), 5+ (Many)
df['bathrooms_category'] = pd.cut(df['bath'], bins=[0, 2, 4, float('inf')], labels=['Few', 'Moderate', 'Many'], right=False)

print('Feature engineering completed.')

# Preparing data for SVM model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

df_cleaned = df.dropna()

# Selecting features and target variable
X = df_cleaned[['total_sqft', 'bath', 'price_per_sqft']]
y = df_cleaned['price']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creating and training the SVM model
svm_model = SVR(kernel='linear')
svm_model.fit(X_train_scaled, y_train)

# Predicting the test set results
y_pred = svm_model.predict(X_test_scaled)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('SVM Model Evaluation:\
MSE:', mse, '\
R2 Score:', r2)