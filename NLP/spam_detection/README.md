# 📬 Spam Detection using NLP

A Natural Language Processing (NLP) project focused on detecting spam messages using traditional machine learning techniques. The project covers everything from exploratory analysis and text preprocessing to vectorization and classification.

---

## 🧾 Problem Statement

With the increasing volume of digital communication, distinguishing between legitimate (ham) and unsolicited (spam) messages has become crucial for both user experience and cybersecurity. This project aims to build an NLP-based spam classifier that can accurately label text messages as **spam** or **ham** using statistical and linguistic features extracted from the text content.

---

## 🧠 Key Concepts Covered

### 🔍 Exploratory Data Analysis (EDA)
- Summary statistics and class distribution
- Word frequency analysis
- N-gram analysis for frequent word patterns in spam messages

### 🧾 DataFrame Processing
- Deriving new columns based on:
  - **Mapping dictionary** (e.g., mapping label text to numeric codes)
  - **`.apply()` function** with custom logic for feature engineering

### 📊 Text Vectorization with `CountVectorizer`
- `fit()` and `transform()` to build the document-term matrix
- Extracting vocabulary using `.get_feature_names_out()`
- Use of **sparse matrix representation** to handle large feature spaces efficiently

### 🧹 Text Preprocessing Techniques
- **Tokenization**
- **Removing stopwords and punctuation** using:
  ```python
  str.maketrans(x, y, z)  # x: chars to replace, y: replacement chars, z: chars to delete
  text.translate(...)     # applies the transformation to text

### ✨ TF-IDF Vectorization
To represent the text numerically and capture the importance of words in context, the project applies the TF-IDF (Term Frequency-Inverse Document Frequency) technique:
- **Term Frequency (TF) :** Counts how many times each word appears in a message.
- **Inverse Document Frequency (IDF) :** Penalizes words that appear frequently across all messages (like "the", "and").
- **L2 Normalization :** Normalizes vectors to unit length to ensure fair comparison regardless of message length.

### 🤖 Classification
A Naive Bayes Classifier was trained on the TF-IDF features to classify messages into spam or ham categories.

**Model highlights:**
- Simple and fast probabilistic classifier based on Bayes' Theorem
- Performs well on high-dimensional text data
- Evaluated using metrics such as:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - AUC score


## Model Results

The following table summarizes the performance of the Naive Bayes classifier on the test dataset:

| Class           | Precision   | Recall     | F1 Score   |
|-----------------|-------------|------------|------------|
| Ham (class 0)   | 0.99        | 0.99       | 0.99       |
| Spam (class 1)  | 0.95        | 0.91       | 0.93       |

### Confusion Matrix
|                  | Predicted: Ham | Predicted: Spam |
|------------------|----------------|-----------------|
| **Actual: Ham**  | 1205           | 8               |
| **Actual: Spam** | 16             | 164             |