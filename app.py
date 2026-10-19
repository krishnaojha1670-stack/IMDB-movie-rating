import streamlit as st
import joblib
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from gensim.utils import simple_preprocess
import os

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }

    .hero-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.25rem;
        text-shadow: 0 0 40px rgba(167, 139, 250, 0.6);
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        text-align: center;
        font-size: 1rem;
        color: #a78bfa;
        margin-bottom: 2.5rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }

    .card {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .result-positive {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.1));
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }

    .result-negative {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(185, 28, 28, 0.1));
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }

    .result-label {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .confidence-text {
        font-size: 1rem;
        color: rgba(255,255,255,0.7);
        margin-top: 0.5rem;
    }

    .meter-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
        margin-top: 1rem;
    }

    .example-tag {
        display: inline-block;
        background: rgba(167, 139, 250, 0.15);
        border: 1px solid rgba(167, 139, 250, 0.3);
        color: #c4b5fd;
        border-radius: 20px;
        padding: 0.3rem 0.9rem;
        font-size: 0.8rem;
        cursor: pointer;
        margin: 0.25rem;
        transition: all 0.2s;
    }

    .stats-row {
        display: flex;
        gap: 1rem;
        justify-content: center;
        margin: 1.5rem 0;
    }

    .stat-box {
        background: rgba(167, 139, 250, 0.1);
        border: 1px solid rgba(167, 139, 250, 0.2);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
        flex: 1;
    }

    .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #a78bfa;
    }

    .stat-label {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    div[data-testid="stTextArea"] textarea {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-size: 1rem !important;
        padding: 1rem !important;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: rgba(167, 139, 250, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.15) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.6) !important;
    }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    base = os.path.dirname(__file__)
    doc_model = Doc2Vec.load(os.path.join(base, "doc2vec_model.model"))
    clf_model = joblib.load(os.path.join(base, "sentiment_classfier.pkl"))
    return doc_model, clf_model

# ─── Predict ───────────────────────────────────────────────────────────────────
def predict_sentiment(review: str, doc_model, clf_model):
    words = simple_preprocess(review)
    if not words:
        return None, None
    vector = doc_model.infer_vector(words, epochs=100)
    predicted = clf_model.predict([vector])[0]
    probab = clf_model.predict_proba([vector])[0]
    confidence = float(max(probab))
    return predicted, confidence

# ─── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎬 Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">IMDB Movie Review Classifier · Doc2Vec + Logistic Regression</div>', unsafe_allow_html=True)

# Model stats
st.markdown("""
<div class="stats-row">
    <div class="stat-box">
        <div class="stat-value">86.9%</div>
        <div class="stat-label">Accuracy</div>
    </div>
    <div class="stat-box">
        <div class="stat-value">50K</div>
        <div class="stat-label">Reviews Trained</div>
    </div>
    <div class="stat-box">
        <div class="stat-value">Doc2Vec</div>
        <div class="stat-label">Embedding</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Loading spinner
with st.spinner("🔄 Loading AI models..."):
    try:
        doc_model, clf_model = load_models()
        st.success("✅ Models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

# Input area
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 📝 Enter a Movie Review")

example_reviews = {
    "😍 Great Movie": "This movie was absolutely fantastic! The acting was superb and the story was gripping from start to finish.",
    "😴 Boring Film": "What a waste of time. Terrible plot, bad acting, and the ending was completely predictable and disappointing.",
    "🎭 Mixed Feelings": "The cinematography was beautiful but the story felt rushed and the characters were poorly developed."
}

col1, col2, col3 = st.columns(3)
selected_example = None
with col1:
    if st.button("😍 Great Movie", key="ex1"):
        selected_example = list(example_reviews.values())[0]
with col2:
    if st.button("😴 Boring Film", key="ex2"):
        selected_example = list(example_reviews.values())[1]
with col3:
    if st.button("🎭 Mixed Feelings", key="ex3"):
        selected_example = list(example_reviews.values())[2]

if selected_example:
    st.session_state["review_input"] = selected_example

review_text = st.text_area(
    label="Review text",
    placeholder="Type or paste your movie review here... (e.g. 'This movie was absolutely amazing!')",
    height=150,
    key="review_input",
    label_visibility="collapsed"
)

word_count = len(review_text.split()) if review_text.strip() else 0
st.caption(f"📊 {word_count} words · Minimum 3 words recommended")
st.markdown('</div>', unsafe_allow_html=True)

# Predict button
if st.button("🔮 Analyze Sentiment", key="predict"):
    if not review_text.strip():
        st.warning("⚠️ Please enter a review to analyze.")
    elif word_count < 2:
        st.warning("⚠️ Please enter at least a few words for a meaningful prediction.")
    else:
        with st.spinner("🧠 Analyzing..."):
            sentiment, confidence = predict_sentiment(review_text, doc_model, clf_model)

        if sentiment is None:
            st.error("Could not process the review. Please try a longer text.")
        else:
            if sentiment == "positive":
                emoji = "🌟"
                label_color = "result-positive"
                sentiment_display = "POSITIVE"
            else:
                emoji = "💔"
                label_color = "result-negative"
                sentiment_display = "NEGATIVE"

            st.markdown(f"""
            <div class="{label_color}">
                <div class="result-label">{emoji} {sentiment_display}</div>
                <div class="confidence-text">Confidence: <strong>{confidence:.1%}</strong></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="meter-label">Confidence Meter</div>', unsafe_allow_html=True)
            st.progress(confidence)

            # Interpretation
            if confidence >= 0.85:
                st.success("🎯 Very high confidence — the model is strongly certain about this prediction.")
            elif confidence >= 0.70:
                st.info("✅ Good confidence — reliable prediction.")
            else:
                st.warning("⚠️ Lower confidence — the review may contain mixed signals.")

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color: rgba(255,255,255,0.3); font-size:0.8rem;'>"
    "Built with Doc2Vec + Logistic Regression · Trained on IMDB 50K Dataset"
    "</p>",
    unsafe_allow_html=True
)
