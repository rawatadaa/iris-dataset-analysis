# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data_path = 'C:/Users/HP/Desktop/telco_customer_churn.csv'  # Path to dataset on your desktop
data = pd.read_csv(data_path)

# Display basic dataset information
print("Dataset Overview:")
print(data.info())
print("\nFirst 5 rows:\n", data.head())

# Step 1: Data Exploration
print("\nSummary Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Visualizations
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x='Churn', palette='viridis')
plt.title("Churn Distribution")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=data, x='Subscription Plan', y='Monthly Charges', hue='Churn', palette='viridis')
plt.title("Monthly Charges by Subscription Plan and Churn")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=data, x='Subscription Plan', y='Total Usage Hours', hue='Churn', palette='viridis')
plt.title("Usage Hours by Subscription Plan and Churn")
plt.show()

# Correlation heatmap
corr = data.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Step 2: Data Cleaning & Preprocessing
# Handle missing values (if any)
for column in data.columns:
    if data[column].isnull().sum() > 0:
        if data[column].dtype == 'object':
            data[column].fillna(data[column].mode()[0], inplace=True)
        else:
            data[column].fillna(data[column].mean(), inplace=True)

# Encode categorical variables and normalize numerical features
categorical_features = ['Gender', 'Subscription Plan']
numerical_features = ['Age', 'Monthly Charges', 'Total Usage Hours', 'Support Tickets']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# Step 3: Model Development
# Split the dataset into training and testing sets
X = data.drop(['Customer ID', 'Churn'], axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with Random Forest Classifier
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', verbose=1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Evaluate the model
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['A qctive', 'Churned'], yticklabels=['Active', 'Churned'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 4: Insights & Recommendations
# Feature importance
feature_names = numerical_features + list(best_model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out())
feature_importances = best_model.named_steps['classifier'].feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title("Feature Importance")
plt.show()

# Recommendations
print("\nBusiness Recommendations:")
print("1. Target customers on lower-tier plans with personalized offers or discounts.")
print("2. Engage customers with low usage hours by promoting content tailored to their interests.")
print("3. Improve customer support to reduce the impact of unresolved issues.")
