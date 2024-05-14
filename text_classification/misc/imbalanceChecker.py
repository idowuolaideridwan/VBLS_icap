import pandas as pd
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you've already loaded the DataFrame
df = pd.read_csv('../data/dataset_v2.csv')

# Initial class distribution
print("Class distribution before resampling:")
print(df['comment_type'].value_counts())

# Visualize the initial class distribution
sns.countplot(x='comment_type', data=df)
plt.title('Class Distribution Before Resampling')
plt.xlabel('Comment Type')
plt.ylabel('Frequency')
plt.show()

# Handle class imbalance
df_majority = df[df['comment_type'] == 'answer']
df_minority = df[df['comment_type'] == 'question']

# Upsample the minority class
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=123)

# Combine majority class with upsampled minority class
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Class distribution after resampling
print("Class distribution after resampling:")
print(df_balanced['comment_type'].value_counts())

# Visualize the class distribution after resampling
sns.countplot(x='comment_type', data=df_balanced)
plt.title('Class Distribution After Resampling')
plt.xlabel('Comment Type')
plt.ylabel('Frequency')
plt.show()
