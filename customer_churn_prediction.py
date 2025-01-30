import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

file_path = 'telco_customer_churn.csv'  
data = pd.read_csv(file_path)

print(data.info())
print(data.describe())
print(data.isnull().sum())

plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Subscription Plan', hue='Churn')
plt.title('Churn Rate by Subscription Plan')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Churn', y='Total Usage Hours')
plt.title('Total Usage Hours by Churn Status')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='Age', hue='Churn', multiple='stack', bins=30)
plt.title('Age Distribution by Churn Status')
plt.show()

if data.isnull().sum().any():
    print("Missing values found. Dropping rows with missing values.")
    data.dropna(inplace=True)

scaler = StandardScaler()
data[['Monthly Charges', 'Total Usage Hours']] = scaler.fit_transform(data[['Monthly Charges', 'Total Usage Hours']])

data = pd.get_dummies(data, columns=['Gender', 'Subscription Plan'], drop_first=True)

X = data.drop(['Customer ID', 'Churn'], axis=1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

feature_importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature')
plt.title('Feature Importance')
plt.show()

joblib.dump(model, 'customer_churn_model.pkl')