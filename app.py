from flask import Flask, render_template, request
import joblib
import re


app = Flask(__name__)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


# ==========================================
# TEXT CLEANING
# ==========================================

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


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================
# PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    news_text = request.form.get("news_text", "")

    if not news_text.strip():

        return render_template(
            "index.html",
            prediction="Please enter some news text."
        )


    # Clean text
    cleaned_news = clean_text(news_text)


    # Convert text to TF-IDF
    news_vector = vectorizer.transform(
        [cleaned_news]
    )


    # Predict
    prediction = model.predict(
        news_vector
    )[0]


    # Probability
    probabilities = model.predict_proba(
        news_vector
    )[0]

    confidence = max(probabilities) * 100


    if prediction == 0:

        result = "FAKE NEWS"

    else:

        result = "REAL NEWS"


    return render_template(
        "index.html",
        prediction=result,
        confidence=round(confidence, 2),
        news_text=news_text
    )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )