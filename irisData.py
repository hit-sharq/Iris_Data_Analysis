import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()

# Create a DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

print("\nBasic Statistics:")
basic_stats = df.describe()
print(basic_stats)

# Group b
grouped_df = df.groupby('species').mean()
print("\nMean of Numerical Features by Species:")
print(grouped_df)


plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
sns.barplot(x=grouped_df.index, y='petal length (cm)', data=grouped_df)
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')

# Histogram of Petal Width
plt.subplot(2, 2, 2)
sns.histplot(df['petal width (cm)'], bins=10, kde=True)
plt.title('Distribution of Petal Width')
plt.xlabel('Petal Width (cm)')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', style='species', s=100)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')

plt.tight_layout()
plt.show()

