import joblib
import re

# Load model
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


# Text cleaning function
def clean_text(text):
    text = str(text)
    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# Test news
news = """
The government announced new economic reforms today
to support small businesses and create more employment.
"""


# Clean news
cleaned_news = clean_text(news)

# Convert to TF-IDF
news_vector = vectorizer.transform([cleaned_news])

# Prediction
prediction = model.predict(news_vector)[0]

# Probability
probabilities = model.predict_proba(news_vector)[0]

print("=" * 60)
print("FAKE NEWS DETECTOR - MODEL TEST")
print("=" * 60)

print("\nPrediction number:", prediction)

print("\nProbabilities:")
print("FAKE:", round(probabilities[0] * 100, 2), "%")
print("REAL:", round(probabilities[1] * 100, 2), "%")


if prediction == 0:
    print("\nRESULT: FAKE NEWS")
else:
    print("\nRESULT: REAL NEWS")

print("=" * 60)