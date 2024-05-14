import os
import joblib
import nltk

# Label mapping dictionary
label_dict = {0: 'non-question', 1: 'question'}

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
    question_words = {'what', 'where', 'when', 'how', 'why', 'did', 'do', 'does', 'have', 'has', 'am', 'is', 'are',
                      'can',
                      'could', 'may', 'would', 'will', 'should', "didn't", "doesn't", "haven't", "isn't", "aren't",
                      "can't", "couldn't", "wouldn't", "won't", "shouldn't", '?'}
    words = set(nltk.word_tokenize(text.lower()))
    if question_words.intersection(words):
        return 'question'
    else:
        return 'non-question'


# Classify a sentence using a logistic regression model with label translation
def classify_sentence(model, text):
    prediction = model.predict([text])[0]
    return label_dict[prediction]  # Translate label


# Sequential filtering classification
def sequential_filtering(models, text):
    # First, try to classify with the rule-based system
    preliminary_result = rule_based_classifier(text)

    # If the result is 'non-question', consider it classified
    if preliminary_result == 'non-question':
        return preliminary_result
    # Otherwise, send to the logistic regression model for further analysis
    else:
        return classify_sentence(models['logistic_regression_model'], text)


# Main function for interaction
def main():
    # nltk.download('punkt')
    models_folder_path = 'models/'
    models = load_models(models_folder_path)

    while True:
        sentence = input("Enter a sentence to classify or type 'exit' to quit: ")
        if sentence.lower() == 'exit':
            break

        classification_result = sequential_filtering(models, sentence)
        print("Classification Result:", classification_result)


if __name__ == "__main__":
    main()
