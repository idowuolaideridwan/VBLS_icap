import os
import joblib
import nltk

def rule_based_classifier(text):
    """ Classify the text based on predefined rules identifying questions. """
    question_words = {'what', 'where', 'when', 'how', 'why', 'did', 'do', 'does', 'have', 'has', 'am', 'is', 'can',
                      'could', 'may', 'would', 'will', 'should', "didn't", "doesn't", "haven't", "isn't", "aren't",
                      "can't", "couldn't", "wouldn't", "won't", "shouldn't", 'how', '?'}
    words = set(nltk.word_tokenize(text.lower()))
    return 'question' if question_words.intersection(words) else 'non-question'

def load_model(model_path):
    """ Load a pre-trained model from the specified file. """
    return joblib.load(model_path)

def logistic_regression_classifier(model, text):
    """ Classify a sentence using the loaded logistic regression model. """
    prediction = model.predict([text])
    return prediction[0]

def hybrid_classifier(models, text):
    """ Determine the classification of text using a hybrid approach. """
    # First use the rule-based classifier
    if rule_based_classifier(text) == 'question':
        return 'question'
    else:
        # Use the logistic regression model
        model_name = 'best_model'
        if model_name in models:
            return logistic_regression_classifier(models[model_name], text)
        else:
            return "Statement"

def main():
    # Path to the folder containing saved models
    models_folder_path = 'models/'

    # Load all .pkl models from the specified folder
    models = {}
    for filename in os.listdir(models_folder_path):
        if filename.endswith('.pkl'):
            model_path = os.path.join(models_folder_path, filename)
            model_name = filename.split('.pkl')[0]
            models[model_name] = load_model(model_path)

    # Input sentence to classify
    sentence = input("Enter a sentence to classify: ")

    classification_result = hybrid_classifier(models, sentence)
    rule_based_result = rule_based_classifier(sentence)
    print("Initial: ", rule_based_result)

    if classification_result == 1:
        print("Classified as an Answer")
    else:
        print("Classified as a Question")

if __name__ == "__main__":
    # nltk.download('punkt')  # Ensure required NLTK resources are downloaded
    main()
