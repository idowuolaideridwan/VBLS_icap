import os
import joblib
import nltk
from sklearn.linear_model import LogisticRegression

# Load all pre-trained models from a specified folder
def load_models(folder_path):
    models = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl'):
            model_name = filename.split('.pkl')[0]
            model_path = os.path.join(folder_path, filename)
            models[model_name] = joblib.load(model_path)
    return models

# Rule-based classifier function
def rule_based_classifier(text):
    question_words = {'what', 'where', 'when', 'how', 'why', 'did', 'do', 'does', 'have', 'has', 'am', 'is', 'are', 'can',
                      'could', 'may', 'would', 'will', 'should', "didn't", "doesn't", "haven't", "isn't", "aren't",
                      "can't", "couldn't", "wouldn't", "won't", "shouldn't", '?'}
    words = set(nltk.word_tokenize(text.lower()))
    return 'question' if question_words.intersection(words) else 'non-question'

# Classify a sentence using a logistic regression model
def classify_sentence(model, text):
    return model.predict([text])[0]

# Majority vote classification combining rule-based and logistic regression
def majority_vote_classification(models, text):
    rule_result = rule_based_classifier(text)
    logistic_result = classify_sentence(models['logistic_regression_model'], text)

    results = [rule_result, logistic_result]
    final_classification = max(set(results), key=results.count)
    return final_classification

# Prepare stacking features from classifiers
def prepare_stacking_features(texts, models):
    features = []
    for text in texts:
        rule_result = 1 if rule_based_classifier(text) == 'question' else 0
        logistic_result = 1 if classify_sentence(models['logistic_regression_model'], text) == 'question' else 0
        features.append([rule_result, logistic_result])
    return features

# Train a stacking classifier on generated features
def train_stacking_classifier(features, labels):
    model = LogisticRegression()
    model.fit(features, labels)
    return model

# Predict using a stacking classifier
def predict_with_stacking(model, features):
    return model.predict(features)

# Main driver function
def main():
    # nltk.download('punkt')
    models_folder_path = 'models/'
    models = load_models(models_folder_path)

    train_texts = ["How are you?", "This is a statement.", "What is the time?"]
    train_labels = ['question', 'non-question', 'question']

    features = prepare_stacking_features(train_texts, models)
    labels = [1 if label == 'question' else 0 for label in train_labels]

    stacking_model = train_stacking_classifier(features, labels)

    sentence = input("Enter a sentence to classify: ")
    print("Majority Vote Classification Result:", majority_vote_classification(models, sentence))
    feature = prepare_stacking_features([sentence], models)
    print("Stacking Classification Result:", "question" if predict_with_stacking(stacking_model, feature)[0] == 1 else "non-question")

if __name__ == "__main__":
    main()
