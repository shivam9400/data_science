Modular Text Classification

Structure:
- src/tokenizers: tokenizer implementations
- src/embeddings: embedding utilities
- src/models: model factory
- src/utils: helpers
- app: Streamlit UI
- deploy: Dockerfile

Quick start:
1. Train a model: python -m src.train --output_dir artifacts
2. Run locally: streamlit run app/streamlit_app.py
3. Build Docker: docker build -t text-classifier -f deploy/Dockerfile .

Notes:
- This is a toy scaffold. Replace dataset loader and add more tokenizers/embeddings as needed.
