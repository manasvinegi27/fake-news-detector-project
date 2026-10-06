import pandas as pd
import re
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("=" * 60)
print("FAKE NEWS DETECTOR - MODEL TRAINING")
print("=" * 60)

dataset_path = "dataset/news.csv"

print("\nLoading dataset...")

df = pd.read_csv(dataset_path)

print("Dataset loaded successfully!")
print("Number of rows:", len(df))
print("Columns:", list(df.columns))


# ==========================================
# 2. CHECK REQUIRED COLUMNS
# ==========================================

if "text" not in df.columns or "label" not in df.columns:
    print("\nERROR: Dataset must contain 'text' and 'label' columns.")
    print("Available columns:", list(df.columns))
    exit()


# ==========================================
# 3. CLEAN DATA
# ==========================================

print("\nCleaning dataset...")

df = df[["text", "label"]].copy()

df["text"] = df["text"].fillna("")
df["label"] = df["label"].fillna("")


# Remove empty rows
df = df[df["text"].str.strip() != ""]
df = df[df["label"].astype(str).str.strip() != ""]


# ==========================================
# 4. TEXT CLEANING FUNCTION
# ==========================================

def clean_text(text):

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML
    text = re.sub(r"<.*?>", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


print("Cleaning text...")

df["text"] = df["text"].apply(clean_text)


# ==========================================
# 5. CONVERT LABELS
# ==========================================

print("\nChecking labels...")

print(df["label"].value_counts())


# Convert labels into 0 and 1
unique_labels = df["label"].astype(str).str.lower().unique()

print("\nUnique labels:", unique_labels)


def convert_label(label):

    label = str(label).strip().lower()

    # Common fake labels
    if label in ["fake", "false", "0"]:
        return 0

    # Common real labels
    if label in ["real", "true", "1"]:
        return 1

    return None


df["label_encoded"] = df["label"].apply(convert_label)

# Remove unknown labels
df = df.dropna(subset=["label_encoded"])

df["label_encoded"] = df["label_encoded"].astype(int)


print("\nFinal label distribution:")
print(df["label_encoded"].value_counts())


# ==========================================
# 6. INPUT AND OUTPUT
# ==========================================

X = df["text"]
y = df["label_encoded"]


# ==========================================
# 7. TRAIN TEST SPLIT
# ==========================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 8. TF-IDF VECTORIZATION
# ==========================================

print("\nCreating TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    max_features=50000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


print("TF-IDF transformation completed.")


# ==========================================
# 9. TRAIN MACHINE LEARNING MODEL
# ==========================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)

print("Model training completed!")


# ==========================================
# 10. TEST MODEL
# ==========================================

print("\nTesting model...")

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["FAKE", "REAL"]
))


# ==========================================
# 11. CREATE MODEL FOLDER
# ==========================================

os.makedirs("model", exist_ok=True)


# ==========================================
# 12. SAVE MODEL
# ==========================================

print("\nSaving model...")

joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\nModel saved successfully!")


# ==========================================
# 13. FINAL MESSAGE
# ==========================================

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nCreated files:")

print("1. model/fake_news_model.pkl")
print("2. model/tfidf_vectorizer.pkl")

print("\nNext step: Create Flask backend using app.py")