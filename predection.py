import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('telco_users.csv')

# print(df.shape) # 10000 rows and 18 columns
# print(df.head()) # first 5 rows
# print(df.dtypes) # data types of each column
# print(df.isnull().sum()) # check for null values

# fill null values with 0
# df = df.fillna(0)

# print(df.isnull().sum())
# print(df['Churn'].unique())
# print(df.head())

# Convert TotalCharges to numeric and handle missing values properly
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(df['TotalCharges'].isnull().sum())

# Fix the chained assignment warning by using a different approach
median_charges = df['TotalCharges'].median()
df.loc[df['TotalCharges'].isnull(), 'TotalCharges'] = median_charges

# Drop customerID column
df = df.drop('customerID', axis=1)

# Convert binary columns to numeric
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Convert gender to numeric
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# Convert Churn to numeric before one-hot encoding
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Apply one-hot encoding to categorical columns
df = pd.get_dummies(df, drop_first=True)

# Training data and predicting REGRESSION
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Increase max_iter to fix convergence warning
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

new_customer = X.iloc[42:43]

predection = model.predict(new_customer)
probability = model.predict_proba(new_customer)

print("Predection: ",predection[0])
print("Probability of churning: ",round(probability[0][1]*100, 2),"%")

# Set style
sns.set(style="whitegrid")

# 1. Churn Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.xticks([0, 1], ['No Churn', 'Churn'])
plt.ylabel("Number of Customers")
plt.show()

# 2. Churn by Contract Type
plt.figure(figsize=(8, 5))
sns.countplot(x='Contract_Two year', hue='Churn', data=df)
plt.title("Churn by Contract Type")
plt.ylabel("Count")
plt.show()

