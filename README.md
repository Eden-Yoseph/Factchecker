# 🧠 TruthLens – Fake News Detection Web App

TruthLens is a lightweight AI-powered web app that detects whether a news paragraph is likely **real or fake**.  
Built in under 3 hours using Python (Flask) and **Hugging Face**'s inference API, it combines pre-trained models with rule-based feature analysis to make smart predictions.

---

## 🚀 Features

- ✅ Paste any news paragraph for analysis
- 🧠 Uses Hugging Face models for:
  - Toxic/misleading language detection
  - Sentiment classification
- 📊 Calculates a fake news probability score
- 🔎 Explains key reasoning in plain English
- 🌐 Web-based frontend (HTML + JS) with Flask backend
- 🔒 Secure token handling with fallback for unauthenticated use

---

## 🧠 Models Used (via Hugging Face)

| Purpose        | Model Name                                                |
|----------------|------------------------------------------------------------|
| Misleading/Toxicity | `martin-ha/toxic-comment-model`                      |
| Sentiment Analysis  | `cardiffnlp/twitter-roberta-base-sentiment-latest`  |
| Fallback / Chat     | `microsoft/DialoGPT-medium`                          |
