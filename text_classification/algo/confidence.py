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

# Classify a sentence using a logistic regression model with confidence
def classify_sentence_with_confidence(model, text):
    probabilities = model.predict_proba([text])
    predicted_class = model.predict([text])[0]
    max_probability = max(probabilities[0])
    return predicted_class, max_probability

# Confidence level routing classifier
def confidence_level_routing(models, text, threshold=0.8):
    predicted_class, confidence = classify_sentence_with_confidence(models['logistic_regression_model'], text)
    if confidence >= threshold:
        return predicted_class
    else:
        return rule_based_classifier(text)

# Main function for interaction
def main():
    # nltk.download('punkt')
    models_folder_path = 'models/'
    models = load_models(models_folder_path)

    while True:
        sentence = input("Enter a sentence to classify or type 'exit' to quit: ")
        if sentence.lower() == 'exit':
            break

        classification_result = confidence_level_routing(models, sentence)
        print("Classification Result:", classification_result)

if __name__ == "__main__":
    main()
