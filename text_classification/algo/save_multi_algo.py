import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess the dataset
df = pd.read_csv('../data/discourse/cleaned_dataset.csv')

# Preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english')).union({'some', 'additional', 'words'})
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    return ' '.join([lemmatizer.lemmatize(word) for word in words if word not in stop_words])

df['comment'] = df['comment'].apply(preprocess_text)

# Handle class imbalance
df_majority = df[df['comment_type'] == 0]
df_minority = df[df['comment_type'] == 1]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=123)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df_balanced['comment'], df_balanced['comment_type'], test_size=0.3, random_state=42)

def build_pipeline(classifier):
    return Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000)),
        ('classifier', classifier)
    ])

classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'SVM': SVC(kernel='linear', probability=True, random_state=42),
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'SGD Classifier': SGDClassifier(random_state=42)
}

def plot_and_save_confusion_matrix(y_true, y_pred, classes, classifier_name):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10,7))
    sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {classifier_name}')
    plt.savefig(f'{classifier_name}_confusion_matrix.png')  # Save the figure
    plt.close()  # Close the plot to free up memory

def plot_and_save_classification_report(y_true, y_pred, classifier_name):
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjust size as needed
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=report_df.values, colLabels=report_df.columns, rowLabels=report_df.index, loc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(8)  # Adjust font size
    the_table.scale(1.2, 1.2)  # Adjust scale
    plt.title(f'Classification Report for {classifier_name}')
    plt.savefig(f'{classifier_name}_classification_report.png')
    plt.close()


for name, classifier in classifiers.items():
    pipeline = build_pipeline(classifier)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Save confusion matrix and classification report images
    plot_and_save_confusion_matrix(y_test, y_pred, classes=['Class1', 'Class2'], classifier_name=name)
    plot_and_save_classification_report(y_test, y_pred, classifier_name=name)
