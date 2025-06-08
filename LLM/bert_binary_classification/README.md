# 🚀 BERT-Based Text Classification

> A text classification pipeline using pretrained BERT (bert-base-uncased) with PyTorch and Hugging Face Transformers.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)

---

## 📊 Overview

> This project builds a binary classification model using BERT to categorize text data. It includes data preprocessing, architecture definition, training with class imbalance handling, evaluation, and inference.

---

## ❓ Problem Statement

> In many NLP applications like sentiment analysis, spam detection, and intent classification, it is crucial to accurately classify text into predefined categories. This project solves the problem by leveraging BERT's deep contextual understanding to build a robust classifier for a labeled text dataset.

---

## 📁 Dataset

- **Source**: ".\dataset\spamdata_v2.csv"
- **Format**: Text + Labels (train, validation, and test splits)
- **Structure**:
  - `train_text` – list of training sentences
  - `train_labels` – corresponding class labels (e.g., 0 or 1)
  - `val_text`, `val_labels` – validation split
  - `test_text`, `test_labels` – test split

---

## 🔍 Approach

Steps followed in the project:

- ✅ Load `bert-base-uncased` and freeze weights
- ✅ Tokenize text using `BertTokenizerFast` with attention masks
- ✅ Convert tokens to PyTorch tensors
- ✅ Handle class imbalance using `compute_class_weight` and weighted `NLLLoss`
- ✅ Build a custom architecture:
  - Dropout, ReLU, Linear layers on top of BERT’s CLS token
- ✅ Train and evaluate the model using:
  - Gradient clipping
  - Best model checkpointing by validation loss
- ✅ Final prediction on test set and report using `classification_report`

---

## ✅ Results

- The model achieved strong performance with proper generalization.
- Class imbalance handled via computed weights.

| Metric       | Score (Example) |
|--------------|------------------|
| Precision_spam    | 0.97            |
| Recall_spam       | 0.86            |
| Precision_ham     | 0.49            |
| Recall_ham        | 0.84            |

---
