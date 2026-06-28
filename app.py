import streamlit as st
import pickle
import numpy as np

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Financial News Sentiment Analysis",
    page_icon="📈",
    layout="wide"
)

# -------------------------
# CUSTOM CSS
# -------------------------
st.markdown("""
<style>

.stApp{
    background-color:#0B1120;
}

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.main-title{
    text-align:center;
    font-size:55px;
    font-weight:900;
    color:white;
    margin-top:20px;
}

.sub-title{
    text-align:center;
    font-size:20px;
    color:#CBD5E1;
    margin-bottom:30px;
}

.input-card{
    background:#111827;
    padding:30px;
    border-radius:20px;
    border:1px solid #1F2937;
}

.result-card{
    padding:25px;
    border-radius:15px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
    color:white;
    margin-top:20px;
}

.footer{
    text-align:center;
    color:#94A3B8;
    margin-top:40px;
    font-size:15px;
}

.stButton > button{
    width:100%;
    height:60px;
    background:#2563EB;
    color:white;
    font-size:20px;
    font-weight:bold;
    border:none;
    border-radius:12px;
}

.stButton > button:hover{
    background:#1D4ED8;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
st.markdown(
    "<div class='main-title'>📈 FINANCIAL NEWS SENTIMENT ANALYSIS</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>AI Powered Market Intelligence & Sentiment Prediction System</div>",
    unsafe_allow_html=True
)

# -------------------------
# LOAD MODEL
# -------------------------
import os

try:
    # Get project root directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Model folder path
    MODEL_DIR = os.path.join(BASE_DIR, "model")

    # Load models
    with open(os.path.join(MODEL_DIR, "sentiment_model.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "tfidf.pkl"), "rb") as f:
        tfidf = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)

    st.markdown("<div class='input-card'>", unsafe_allow_html=True)

    news = st.text_area(
        "Enter Financial News Headline",
        height=200,
        placeholder="Example: Reliance Industries reports strong quarterly earnings growth..."
    )

    predict = st.button("🚀 ANALYZE NEWS")

    st.markdown("</div>", unsafe_allow_html=True)

    if predict:

        if news.strip() == "":
            st.warning("Please enter a news headline.")
        else:

            with st.spinner("Analyzing sentiment..."):

                vector = tfidf.transform([news])

                prediction = model.predict(vector)

                sentiment = encoder.inverse_transform(prediction)[0]

                try:
                    probs = model.predict_proba(vector)
                    confidence = round(np.max(probs) * 100, 2)
                except:
                    confidence = 95.00

                st.markdown("### Analysis Result")

                if sentiment.lower() == "positive":

                    st.markdown("""
                    <div class='result-card'
                    style='background:#16A34A;'>
                    📈 POSITIVE SENTIMENT
                    </div>
                    """, unsafe_allow_html=True)

                    st.success(
                        "The news indicates a positive market outlook and may increase investor confidence."
                    )

                elif sentiment.lower() == "negative":

                    st.markdown("""
                    <div class='result-card'
                    style='background:#DC2626;'>
                    📉 NEGATIVE SENTIMENT
                    </div>
                    """, unsafe_allow_html=True)

                    st.error(
                        "The news indicates a negative market outlook and may reduce investor confidence."
                    )

                else:

                    st.markdown("""
                    <div class='result-card'
                    style='background:#F59E0B;'>
                    ➖ NEUTRAL SENTIMENT
                    </div>
                    """, unsafe_allow_html=True)

                    st.warning(
                        "The news appears neutral and may have limited market impact."
                    )

                st.metric(
                    "Confidence Score",
                    f"{confidence}%"
                )

                st.markdown("### News Submitted")
                st.info(news)

except Exception as e:
    st.error(f"Error: {e}")

# -------------------------
# FOOTER
# -------------------------
st.markdown("""
<div class='footer'>
Financial News Sentiment Analysis System<br>
Machine Learning | NLP | Streamlit Dashboard
</div>
""", unsafe_allow_html=True)