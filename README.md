# IMDB Sentiment Analyzer 🎬

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | **86.89%** |
| Embedding | Doc2Vec (100-dim, 100 epochs) |
| Classifier | Logistic Regression |
| Training data | 40,000 reviews |
| Test data | 10,000 reviews |

---

## 🏗️ Project Structure

```
project/
│
├── app.py                    # Streamlit web app
├── requirements.txt          # Python dependencies
├── doc2vec_model.model       # Trained Doc2Vec model
├── sentiment_classfier.pkl   # Trained Logistic Regression model
├── .streamlit/
│   └── config.toml           # Streamlit theme & server config
├── data/
│   └── IMDB Dataset.csv      # Raw training data
└── project.ipynb             # Training notebook
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/imdb-sentiment-analyzer.git
cd imdb-sentiment-analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

App opens at → `http://localhost:8501`

---

## ☁️ Deploy to Streamlit Cloud

link :- https://imdb-movie-rating-ah8kpfolgxqmepvrqji2wy.streamlit.app/

---

## 📦 Dependencies

- `streamlit` — Web UI
- `gensim` — Doc2Vec model
- `scikit-learn` — Logistic Regression
- `joblib` — Model serialization
- `numpy` — Vector operations

---

*Built with ❤️ · Trained on IMDB Dataset*
owner 
krishna ojha
