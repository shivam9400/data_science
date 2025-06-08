# 🤗 Learning LLMs with Hugging Face – Chapters 1 to 3

This folder contains my implementation notebook based on the **first three chapters** of the [Hugging Face LLM Course](https://huggingface.co/learn/llm-course). It covers foundational concepts in large language models (LLMs), how to use them with Hugging Face tools, and how to fine-tune them on custom datasets.

All concepts are documented in a single, well-commented Jupyter notebook with practical examples and explanations.

---

## 📘 Major Topics Covered

### ✅ [Chapter 1: What is a Language Model?](https://huggingface.co/learn/llm-course/chapter1)
- What is a language model?
- How do Transformer work?
- Transformer architectures

### ✅ [Chapter 2: Using a Language Model](https://huggingface.co/learn/llm-course/chapter2)
- Introduction to the `transformers` library
- Using the `pipeline()` API for:
  - Text generation
  - Text classification
- Tokenization: how inputs are converted into model-understandable format
- Loading pretrained models and tokenizers from Hugging Face Hub

### ✅ [Chapter 3: Fine-Tuning a Language Model](https://huggingface.co/learn/llm-course/chapter3)
- Introduction to the `datasets` library
- Dataset preprocessing and tokenization
- Training a language model using `Trainer`
- Training a language model **without** using `Trainer`
- Evaluation using `evaluate` library (accuracy, F1, etc.)

---

## 📓 What’s Inside the Notebook
- Code walkthroughs aligned with the official Hugging Face course
- Step-by-step annotated implementations
- Visual explanations and tokenization demos
- Hands-on model training and evaluation
- My personal notes and learnings

---

## 🛠️ Tech Stack
- Python 3.x
- [Transformers](https://github.com/huggingface/transformers)
- [Datasets](https://github.com/huggingface/datasets)
- Evaluate
- PyTorch
- Jupyter Notebook