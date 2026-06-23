import streamlit as st
import pickle

st.set_page_config(
    page_title="Financial News Sentiment Analysis",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Financial News Sentiment Analysis")
st.write("Enter a financial news headline and predict market sentiment.")

try:
    # Load model files
    model = pickle.load(open("sentiment_model.pkl", "rb"))
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    encoder = pickle.load(open("label_encoder.pkl", "rb"))

    st.success("Model loaded successfully!")

    news = st.text_area("Enter Financial News")

    if st.button("Predict Sentiment"):

        if news.strip() == "":
            st.warning("Please enter financial news.")
        else:
            vector = tfidf.transform([news])

            prediction = model.predict(vector)

            result = encoder.inverse_transform(prediction)

            st.success(f"Predicted Sentiment: {result[0]}")

except Exception as e:
    st.error(f"Error loading application: {e}")