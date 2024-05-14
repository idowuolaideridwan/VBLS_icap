import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk

# Ensure NLTK resources are downloaded
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
df = pd.read_csv('../data/dataset_v2.csv')
df['comment'] = df['comment'].apply(preprocess_text)

# Handle class imbalance
df_majority = df[df['comment_type'] == 'answer']
df_minority = df[df['comment_type'] == 'question']
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=123)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df_balanced['comment'], df_balanced['comment_type'], test_size=0.20, random_state=42)

# Define classifier models
classifiers = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Support Vector Machine": SVC(kernel='linear'),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "SGD Classifier": SGDClassifier(loss='hinge')
}

# Loop through classifiers
results = {}
for name, clf in classifiers.items():
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000)),
        ('classifier', clf)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"Accuracy of {name}: {accuracy:.2f}")