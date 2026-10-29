
import re
import joblib

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

# Create FastAPI application
app = FastAPI(
    title="IMDb Sentiment Analysis API",
    description="Predict positive or negative sentiment from movie reviews",
    version="1.0.0"
)

# Load the trained model and TF-IDF vectorizer
model = joblib.load("imdb_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# Text preprocessing function
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Define request format
class ReviewRequest(BaseModel):
    review: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description="Movie review text"
    )


# Define response format
class SentimentResponse(BaseModel):
    review: str
    sentiment: Literal["Positive", "Negative"]
    label: int


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to the IMDb Sentiment Analysis API",
        "docs": "/docs",
        "prediction_endpoint": "/predict"
    }


# Health check endpoint
@app.get("/health")
def health():
    return {"status": "healthy"}


# Sentiment prediction endpoint
@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: ReviewRequest):

    # Clean the input review
    cleaned_review = clean_text(request.review)

    # Convert text into TF-IDF features
    review_vector = vectorizer.transform([cleaned_review])

    # Predict sentiment
    prediction = int(model.predict(review_vector)[0])

    # Convert numeric label to readable sentiment
    sentiment = "Positive" if prediction == 1 else "Negative"

    return {
        "review": request.review,
        "sentiment": sentiment,
        "label": prediction
    }

