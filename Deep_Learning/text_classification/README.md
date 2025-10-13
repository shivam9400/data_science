# Modular Text Classification
This project implements a deep learning model for binary text classification (e.g., sentiment analysis) using a combined Convolutional Neural Network (Conv1D) and Long Short-Term Memory (LSTM) architecture, organized into a modular structure for clarity and scalability.

## Project Structure:
- src/tokenizers: tokenizer implementations
- src/embeddings: embedding utilities
- src/models: model factory for defining and constructing various neural network architectures (e.g., 'cnn_lstm').
- src/utils: helpers
- deploy: Dockerfile
- artifacts: (generated) stores trained models, tokenizers, and configuration files.

## Quick start:
1. Train a model: python -m src.train --output_dir artifacts
2. Run locally: streamlit run streamlit_app.py
3. Build Docker: docker build -t text-classifier -f deploy/Dockerfile .
  a. docker run -p 8501:8501 text-classifier 

### Notes:
- Small Batch Size Warning: The default batch_size=2 is extremely small and may lead to noisy gradients and unstable training. Increase this value (e.g., to 32 or 64) for production environments.

## Deep Learning Model Architecture and Flow
The default architecture (```cnn_lstm```) is a sequential model designed for robust text classification.

### Architecture Specification

| **Layer**           | **Hyperparameter(s)**                      | **Output**                  | **Purpose**                                                                 |
|---------------------|--------------------------------------------|-----------------------------|------------------------------------------------------------------------------|
| **Embedding**       | `embedding_dim = 50`                       | Sequence of 50-dim vectors  | Converts word indices into dense vector representations                     |
| **Conv1D**          | `filters = 128`, `kernel_size = 5`         | 128 local features          | Extracts local patterns (n-grams) from the text                             |
| **MaxPooling1D**    | `pool_size = 2`                            | Downsampled sequence        | Reduces sequence length and focuses on dominant features                    |
| **LSTM**            | `units = 64`                               | Final state vector (64-dim) | Models sequence dependencies and summarizes contextual information          |
| **Dense (Sigmoid)** | `units = 1`                                | Probability score [0, 1]    | Outputs final binary classification score using Sigmoid activation          |

### Data Flow and Matrix Dimensions (Batch Size=2)
The table below traces the data shape through the pipeline using default settings (max_len=20, embedding_dim=50, Conv1D filters=128, LSTM units=64).

| **Layer / Step**         | **Input Shape**     | **Output Shape**     | **Interpretation**                                                                 |
|--------------------------|---------------------|-----------------------|-------------------------------------------------------------------------------------|
| **Input → Padding**      | Text                | `(2, 20)`             | 2 sentences, each padded to 20 token indices                                        |
| **Embedding(50)**        | `(2, 20)`           | `(2, 20, 50)`         | Each token is mapped to a 50-dimensional dense vector                              |
| **Conv1D(128, k=5)**     | `(2, 20, 50)`       | `(2, 16, 128)`        | Extracts 128 features per step; sequence length reduced due to kernel size         |
| **MaxPooling1D(2)**      | `(2, 16, 128)`      | `(2, 8, 128)`         | Downsamples sequence length by a factor of 2                                       |
| **LSTM(64)**             | `(2, 8, 128)`       | `(2, 64)`             | Outputs a 64-dimensional context vector summarizing the sequence                   |

## Technical details
### LSTM Unit Definition
The LSTM(64) layer contains 64 parallel memory units. Each of these units works simultaneously, processing the sequence step-by-step and using its internal Forget, Input, and Output gates to manage long-term context.

### Embedding Trainability
The Embedding layer is trainable=False when initialized with an external matrix (even a random one, as per the train.py logic). This means the randomly initialized word vectors will be fixed throughout the training process and will not be updated to optimize performance. To enable learning, you must remove trainable=False from the Embedding layer definition in the ModelFactory.

### Classification Threshold
The final prediction is determined by a classification threshold (τ) applied to the Dense layer's probability output P. By default, τ=0.5. Changing this threshold directly alters the model's bias towards predicting the positive class (1):
1. Higher τ (e.g., 0.8) leads to higher Precision.
2. Lower τ (e.g., 0.2) leads to higher Recall.