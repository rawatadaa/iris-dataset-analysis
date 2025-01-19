import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris_data = load_iris()
iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
iris_df['target'] = iris_data.target

iris_df['target_name'] = iris_df['target'].apply(lambda x: iris_data.target_names[x])

print("6 few rows of the dataset:")
print(iris_df.head())

print("\nMissing values:")
print(iris_df.isnull().sum())

iris_df.columns = [col.replace(" (cm)", "").replace(" ", "_") for col in iris_df.columns]

print("\nDuplicate rows:", iris_df.duplicated().sum())

print("\nData types:")
print(iris_df.dtypes)

print("\nDescriptive Statistics:")
desc_stats = iris_df.describe()
print(desc_stats)

print("\nMedian:")
print(iris_df.median())

print("\nStandard Deviation:")
print(iris_df.std())

iris_df.iloc[:, :-2].hist(bins=int(np.sqrt(len(iris_df))), figsize=(10, 8))
plt.suptitle('Feature Distributions', y=0.95)
plt.show()

sns.pairplot(iris_df, hue='target_name', diag_kind='hist')
plt.suptitle('Pairwise Scatter Plots', y=1.02)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=iris_df, x='target_name', y='sepal_length')
plt.title('Sepal Length Distribution by Species')
plt.show()

corr_matrix = iris_df.iloc[:, :-2].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

for feature in iris_df.columns[:-2]:
    Q1 = iris_df[feature].quantile(0.25)
    Q3 = iris_df[feature].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((iris_df[feature] < (Q1 - 1.5 * IQR)) | (iris_df[feature] > (Q3 + 1.5 * IQR))).sum()
    print(f"Outliers in {feature}: {outliers}")

    plt.figure(figsize=(6, 4))
    sns.boxplot(x=iris_df[feature])
    plt.title(f'Outliers in {feature}')
    plt.show()


