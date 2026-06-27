# Arabic Sentiment Analysis

Classifying Arabic tweets as Positive or Negative using an LSTM deep learning model trained on a large Arabic Twitter corpus.

---

## Problem Statement
Arabic NLP is an underrepresented field in machine learning due to the complexity of the language — multiple dialects, diacritics, and informal social media writing styles make Arabic text classification significantly harder than English. This project builds a sentiment classifier capable of handling real-world Arabic tweets across different dialects and writing styles.

---

## Dataset
- **Source:** [Arabic Sentiment Twitter Corpus — Kaggle](https://www.kaggle.com/datasets/mksaad/arabic-sentiment-twitter-corpus)
- **Size:** ~45,000 tweets split across 4 TSV files
- **Labels:** Positive (pos) / Negative (neg)
- **Split:** Pre-divided into train and test sets

---

## Workflow
1. Loading data from 4 separate TSV files — positive and negative for train and test
2. Text cleaning — removed URLs, mentions, hashtags, tashkeel, emojis, and non-Arabic characters
3. Label encoding — converted pos/neg to 1/0
4. Tokenization — built a vocabulary of 10,000 most common Arabic words
5. Padding — fixed all sequences to 50 tokens
6. Built and trained LSTM model with Early Stopping to prevent overfitting
7. Evaluated with accuracy, confusion matrix, and classification report

---

## Model Architecture
| Layer | Details |
|-------|---------|
| Embedding | 10,000 vocab, 64 dimensions |
| LSTM | 64 units |
| Dropout | 50% |
| Dense | 1 unit, sigmoid activation |

---

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | 77% |
| Negative F1 | 0.79 |
| Positive F1 | 0.76 |
| Macro F1 | 0.77 |

---

## Confusion Matrix Summary
| | Predicted Negative | Predicted Positive |
|--|--|--|
| **Actual Negative** | 4814 ✅ | 954 ❌ |
| **Actual Positive** | 1664 ❌ | 4088 ✅ |

---

## Key Findings
- The model performs better at detecting **negative tweets** (recall 83%) than positive ones (recall 71%)
- Negative Arabic language tends to be more distinct and easier to classify
- The model handles mixed sentiment tweets reasonably well
- Early stopping prevented overfitting — best results achieved at epoch 2

---

## Challenges
- Arabic has multiple dialects — Egyptian, Gulf, Levantine — all written differently
- Heavy use of emojis, slang, and abbreviations on Twitter
- Tashkeel (diacritics) adds noise that needed to be removed during preprocessing

---

## Libraries Used
- Pandas, NumPy
- Matplotlib, Seaborn
- TensorFlow, Keras
- Scikit-learn
- re
