from transformers import pipeline

def classify_sentiment(comments):
    # Load the sentiment analysis pipeline
    sentiment_pipeline = pipeline("sentiment-analysis")

    # Classify the sentiment of each comment
    results = sentiment_pipeline(comments)

    return results


# Example comments
comments = [
    "If you try to sum up the note revisions later tonight, It will help because I understand it",
    "Continue, I will read it tonight."
]

# Classify the sentiment of comments
sentiment_results = classify_sentiment(comments)

for comment, sentiment in zip(comments, sentiment_results):
    print(f"Comment: '{comment}'\nSentiment: {sentiment['label']}, Score: {sentiment['score']:.2f}\n")
