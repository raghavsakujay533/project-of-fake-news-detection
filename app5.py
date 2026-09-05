import streamlit as st
import joblib
import re
import string

# Page Configuration
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# Load Model
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# Text Cleaning
def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


# Sidebar
st.sidebar.title("📰 Fake News Detector")

st.sidebar.info(
"""
### Features

✔ Detect Fake News

✔ Confidence Score

✔ Word Counter

✔ Character Counter

✔ Fast Prediction
"""
)

st.sidebar.success("Machine Learning Project")


# Title
st.title("📰 Fake News Detection using Machine Learning")

st.write(
"""
Enter the news article below and click **Predict**.
"""
)

news = st.text_area(
    "Paste News Article",
    height=250,
    placeholder="Paste complete news article here..."
)

col1, col2 = st.columns(2)

with col1:
    word_count = len(news.split())

with col2:
    char_count = len(news)

st.metric("Words", word_count)
st.metric("Characters", char_count)

if st.button("🔍 Detect News", use_container_width=True):

    if news.strip() == "":
        st.warning("Please enter a news article.")

    else:

        cleaned = clean(news)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]

        probability = model.predict_proba(vector)

        confidence = max(probability[0]) * 100

        st.divider()

        if prediction == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")

        st.progress(int(confidence))

        st.write(f"### Confidence : {confidence:.2f}%")

        st.subheader("Prediction Probability")

        st.write(
            {
                "Fake": round(probability[0][0]*100,2),
                "Real": round(probability[0][1]*100,2)
            }
        )


st.divider()

st.caption("Developed using Python, Scikit-learn, TF-IDF and Streamlit")