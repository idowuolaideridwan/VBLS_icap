from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Example sentences
sentences = [
    "This method improves accuracy",  # Statement
    "Exploring the implications of this discovery",  # Implicit question or inquiry
    "The results were unexpected",  # Statement
    "Considering alternative approaches for the analysis",  # Implicit question or inquiry
]

# Labels: 0 for Statement, 1 for Implicit Question/Inquiry
labels = np.array([0, 1, 0, 1])

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.25, random_state=42)

# Vectorizing text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Training a simple classifier (logistic regression in this case)
classifier = LogisticRegression()
classifier.fit(X_train_vectors, y_train)

# Predicting and evaluating
predictions = classifier.predict(X_test_vectors)
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy}")

def predict_sentence(sentence, vectorizer, classifier):
    # Vectorize the sentence
    sentence_vector = vectorizer.transform([sentence])

    # Predict using the classifier
    prediction = classifier.predict(sentence_vector)

    # Return the prediction
    return "Question" if prediction[0] == 1 else "Answer"

# Example Usage
sentence = "Considering alternative approaches for the analysis"
prediction = predict_sentence(sentence, vectorizer, classifier)
print(f"Predicted Category: {prediction}")
