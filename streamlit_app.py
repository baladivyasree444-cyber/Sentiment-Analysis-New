
import re
import joblib
import streamlit as st

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# --------------------------------------------------
# 2. Load the trained model and vectorizer
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("imdb_sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer


try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Please place "
        "'imdb_sentiment_model.pkl' and "
        "'tfidf_vectorizer.pkl' in the project folder."
    )
    st.stop()

# --------------------------------------------------
# 3. Text preprocessing
# --------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# --------------------------------------------------
# 4. Application UI
# --------------------------------------------------
st.title("🎬 IMDb Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review below to predict whether "
    "its sentiment is positive or negative."
)

st.info(
    "Machine Learning Model: Logistic Regression\n\n"
    "Text Features: TF-IDF"
)

# Input review
review = st.text_area(
    "Enter your movie review:",
    placeholder=(
        "Example: The movie was fantastic, "
        "with excellent acting and a wonderful story."
    ),
    height=180,
    max_chars=20000
)

# --------------------------------------------------
# 5. Prediction
# --------------------------------------------------
if st.button("Analyze Sentiment", type="primary"):

    if not review.strip():
        st.warning("Please enter a movie review first.")

    else:
        with st.spinner("Analyzing your review..."):

            cleaned_review = clean_text(review)

            review_vector = vectorizer.transform(
                [cleaned_review]
            )

            prediction = int(
                model.predict(review_vector)[0]
            )

            sentiment = (
                "Positive" if prediction == 1
                else "Negative"
            )

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("😊 Positive Sentiment")
        else:
            st.error("😞 Negative Sentiment")

        st.write(f"**Predicted label:** {prediction}")

        with st.expander("View submitted review"):
            st.write(review)

        st.caption(
            "This is a machine learning prediction and "
            "may not correctly classify every review."
        )

# --------------------------------------------------
# 6. Example reviews
# --------------------------------------------------
st.divider()
st.subheader("Try Example Reviews")

if st.button("Use positive example"):
    st.session_state["example_review"] = (
        "This movie was amazing! The acting was "
        "excellent and the story was wonderful."
    )
    st.rerun()

if st.button("Use negative example"):
    st.session_state["example_review"] = (
        "This movie was boring and disappointing. "
        "The story was weak and the acting was poor."
    )
    st.rerun()

if "example_review" in st.session_state:
    st.caption(
        "Example loaded. Copy it into the review box "
        "above and click Analyze Sentiment."
    )
    st.code(st.session_state["example_review"])
