import streamlit as st
import pickle

# Function to load the model and vectorizer
def load_model_and_vectorizer():
    model = pickle.load(open('trained_model.sav', 'rb'))
    vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
    return model, vectorizer

# Function for stemming (assuming you have the stemming function defined)
def stemming(text):
    # Add your stemming code here (e.g., using NLTK or other libraries)
    return text  # Example placeholder, replace with actual stemming logic

# Function for predicting sentiment of custom text
def predict_sentiment(text):
    loaded_model, vectorizer = load_model_and_vectorizer()

    stemmed_text = stemming(text)
    text_vectorized = vectorizer.transform([stemmed_text])
    prediction = loaded_model.predict(text_vectorized)
    
    sentiment_class = int(prediction[0])
    print("prediciton:::")
    pring(sentiment_class)
    # Map prediction to sentiment labels
    sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    return sentiment_mapping.get(sentiment_class, "Unknown")

# Streamlit UI
st.title("Twitter Sentiment Analysis")
st.write("""
    This app predicts the sentiment of **Twitter tweets**.
    Enter a tweet below, and the app will classify it as:
    - **Negative** 
    - **Neutral**
    - **Positive** 
""")

# Text input field for the Twitter tweet
input_text = st.text_area("Enter a Twitter tweet for sentiment analysis:")

# Button to predict sentiment
if st.button("Predict Sentiment"):
    if input_text:
        sentiment = predict_sentiment(input_text)
        st.write(f"Sentiment: {sentiment}")
    else:
        st.write("Please enter a tweet to analyze.")
