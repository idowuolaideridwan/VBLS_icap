from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def load_model(model_name="unitary/toxic-bert"):
    """
    Load the pre-trained model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def classify_comment(comment, tokenizer, model):
    """
    Classify a comment as appropriate or inappropriate.
    """
    inputs = tokenizer(comment, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.sigmoid(logits).squeeze()  # Convert logits to probabilities
    is_inappropriate = probabilities[0] > 0.5  # Threshold can be adjusted based on needs
    return "Inappropriate" if is_inappropriate else "Appropriate"

# Load the model and tokenizer
tokenizer, model = load_model()

# Example comment
comment = "Idiot, That was a wrong answer"

# Classify the comment
classification = classify_comment(comment, tokenizer, model)
print(f"Classification: {classification}")
