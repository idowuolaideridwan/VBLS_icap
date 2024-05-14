import os
import joblib

def load_models(folder_path):
    """ Load all pre-trained models from the specified folder. """
    models = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl'):
            model_name = filename.split('_model.pkl')[0]
            model_path = os.path.join(folder_path, filename)
            model = joblib.load(model_path)
            models[model_name] = model
    return models


def classify_sentence(model, sentence):
    """ Classify a sentence using the loaded model. """
    prediction = model.predict([sentence])
    return prediction[0]


def main():
    # Path to the folder containing saved models
    models_folder_path = 'pkl/'

    # Load all models from the specified folder
    models = load_models(models_folder_path)

    while True:
        # Input sentence to classify
        sentence = input("Enter a sentence to classify (type 'quit' to exit): ")

        if sentence.lower() == 'quit':
            break

        # Classify the given sentence using each loaded model
        for model_name, model in models.items():
            predicted_class = classify_sentence(model, sentence)
            if predicted_class == 1:
                print(f"The input sentence is classified as a question by the {model_name} model.")
            else:
                print(f"The input sentence is classified as an answer by the {model_name} model.")

        # Ask if user wants to classify another sentence
        choice = input("Do you want to classify another sentence? (yes/no): ")
        if choice.lower() != 'yes':
            break


if __name__ == "__main__":
    main()
