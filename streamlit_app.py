import streamlit as st
import pickle
import nltk
import re

# Download NLTK resources (run only once)
nltk.download('punkt')

# Initialize stemmer
stemmer = PorterStemmer()

# Function to load the model and vectorizer (Lazy Loading)
@st.cache_resource
def load_model_and_vectorizer():
    model = pickle.load(open('trained_model.sav', 'rb'))
    vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
    return model, vectorizer


def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation & special characters
    return text


# Function for predicting sentiment of custom text
def predict_sentiment(text):
    loaded_model, vectorizer = load_model_and_vectorizer()

    stemmed_text = clean_text(text)
    text_vectorized = vectorizer.transform([stemmed_text])
    prediction = loaded_model.predict(text_vectorized)

    # Handle cases where prediction is not an integer
    try:
        sentiment_class = int(prediction[0])
    except ValueError:
        st.write(f"Unexpected Prediction: {prediction[0]}")
        sentiment_class = "Unknown"

    # Map prediction to sentiment labels
    sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    return sentiment_mapping.get(sentiment_class, "Unknown")

# Streamlit UI
st.title("📊 Twitter Sentiment Analysis")
st.write("""
    This app predicts the sentiment of **Twitter tweets**.
    Enter a tweet below, and the app will classify it as:
    - **Negative** 😠
    - **Neutral** 😐
    - **Positive** 😊
""")

# Text input field for the Twitter tweet
input_text = st.text_area("✍️ Enter a Twitter tweet for sentiment analysis:")

# Button to predict sentiment
if st.button("🔍 Predict Sentiment"):
    if input_text:
        sentiment = predict_sentiment(input_text)
        st.success(f"📝 Sentiment: **{sentiment}**")
    else:
        st.warning("⚠️ Please enter a tweet to analyze.")
