# 🧠 CNN From Scratch in Python

This repository contains a complete implementation of a **Convolutional Neural Network (CNN)** from scratch using only **NumPy** — no deep learning frameworks like TensorFlow or PyTorch are used.

The goal of this project is to **demystify CNNs** by building them step-by-step, enabling learners to understand what happens under the hood during forward and backward passes.

---

## 🚀 Features

- ✅ Manual implementation of:
  - Convolution layer
  - Max Pooling
  - Flatten layer
  - Fully Connected (Dense) layers
  - Softmax and cross-entropy loss
- ✅ Backpropagation from scratch
- ✅ Training loop built manually
- ✅ No external ML libraries (only NumPy)

---

## 📓 Notebook Overview

| Section | Description |
|--------|-------------|
| **1. Convolution Layer** | Implements 2D convolution with customizable kernel size and stride. |
| **2. Max Pooling Layer** | Downsamples feature maps, retaining dominant features. |
| **3. Flatten + Dense** | Converts 2D data to vector and applies learnable weights. |
| **4. Softmax + Loss** | Applies softmax activation and computes cross-entropy loss. |
| **5. Backpropagation** | Derives and applies gradients to update weights. |
| **6. Training Loop** | Trains model with forward and backward passes. |

---

## 📊 Dataset

This implementation supports grayscale image datasets such as:

- MNIST (via `keras.datasets`) --> `keras.datasets.fashion_mnist.load_data()`