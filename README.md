# IMDB Sentiment Analyzer 🎬

A movie review sentiment classifier trained on the **IMDB 50K Dataset** using **Doc2Vec embeddings** and **Logistic Regression**.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

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

1. Push this repo to GitHub (make sure model files are committed)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → select your repo → set `app.py` as the main file
4. Click **Deploy** 🚀

> **Note:** The `doc2vec_model.model` file is ~62 MB. GitHub allows files up to 100 MB, so it fits without Git LFS. If you get issues, use Git LFS.

---

## 🧠 How It Works

```
Review Text
    │
    ▼
[simple_preprocess]  ← tokenize & lowercase
    │
    ▼
[Doc2Vec.infer_vector]  ← convert to 100-dim vector
    │
    ▼
[LogisticRegression.predict]  ← classify
    │
    ▼
positive / negative + confidence score
```

---

## 📦 Dependencies

- `streamlit` — Web UI
- `gensim` — Doc2Vec model
- `scikit-learn` — Logistic Regression
- `joblib` — Model serialization
- `numpy` — Vector operations

---

*Built with ❤️ · Trained on IMDB Dataset*
