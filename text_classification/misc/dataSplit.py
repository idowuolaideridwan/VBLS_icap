import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import resample
import seaborn as sns

# Assuming you've already loaded the DataFrame
df = pd.read_csv('../data/dataset_v2.csv')

# Handle class imbalance
df_majority = df[df['comment_type'] == 'answer']
df_minority = df[df['comment_type'] == 'question']
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=123)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Split the dataset
X_train, X_temp, y_train, y_temp = train_test_split(df_balanced['comment'], df_balanced['comment_type'], test_size=0.40, stratify=df_balanced['comment_type'], random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

# Assuming 'y_train', 'y_val', 'y_test' are your target arrays or lists
df_visual = pd.DataFrame({
    'Set': ['Train'] * len(y_train) + ['Validation'] * len(y_val) + ['Test'] * len(y_test),
    'Class': list(y_train) + list(y_val) + list(y_test)
})

# Plotting the stacked bar chart
plt.figure(figsize=(10, 6))
sns.countplot(x='Set', hue='Class', data=df_visual)
plt.title('Class Distribution Across Dataset Splits')
plt.xlabel('Dataset Splits')
plt.ylabel('Count')
plt.legend(title='Comment Type')
plt.show()