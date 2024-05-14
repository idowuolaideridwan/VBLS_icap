import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from joblib import dump, load

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Setup enhanced stopwords list
additional_stopwords = {'some', 'additional', 'words'}
stop_words = set(stopwords.words('english')).union(additional_stopwords)

# Setup the lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Load and preprocess the dataset
df = pd.read_csv('../data/discourse/dataset_v2.csv')
df['comment'] = df['comment'].apply(preprocess_text)

# Handle class imbalance
df_majority = df[df['comment_type'] == "question"]   # question
df_minority = df[df['comment_type'] == "answer"]   # answer
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=123)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Split the dataset
X_train, X_temp, y_train, y_temp = train_test_split(df_balanced['comment'], df_balanced['comment_type'], test_size=0.40, stratify=df_balanced['comment_type'], random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

# Set up pipeline with vectorizer and model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('clf', LogisticRegression(random_state=42))
])

# Hyperparameter grid for tuning
param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf__min_df': [1, 5, 10],
    'tfidf__max_df': [0.5, 0.75, 1.0],
    'tfidf__use_idf': [True, False],
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__solver': ['liblinear', 'lbfgs', 'saga']
}

# Set up GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(df_balanced['comment'], df_balanced['comment_type'])

# Save the best model
dump(grid_search.best_estimator_, 'discourse_dataset_model.pkl')

# Load the model (optional, to demonstrate how to reload it)
loaded_model = load('discourse_dataset_model.pkl')

# Best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Evaluate the model using the best estimator from grid search
y_pred = loaded_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
