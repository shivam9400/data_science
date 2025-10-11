import streamlit as st
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.tokenizers.keras_tokenizer import KerasTokenizerWrapper
from src.utils.io import load_json

st.set_page_config(page_title='Text Classifier', layout='centered')

st.title('Modular Text Classification')

artifact_dir = st.sidebar.text_input('Artifacts directory', value='artifacts')

if st.sidebar.button('Load model'):
    model_path = Path(artifact_dir) / 'model.h5'
    tokenizer_path = Path(artifact_dir) / 'tokenizer.json'
    config_path = Path(artifact_dir) / 'config.json'
    if not model_path.exists():
        st.error('Model not found in artifacts. Train and save a model first.')
    else:
        model = load_model(model_path)
        st.success('Model loaded')
        config = load_json(config_path)
        tokenizer = KerasTokenizerWrapper(num_words=config.get('vocab_size', 1000))
        tokenizer.load(tokenizer_path)

        text = st.text_area('Enter text to classify')
        threshold = st.slider('Threshold', 0.0, 1.0, 0.5)
        if st.button('Classify'):
            seq = tokenizer.texts_to_sequences([text])
            pad = pad_sequences(seq, maxlen=config.get('max_len', 20), padding='post')
            prob = float(model.predict(pad, verbose=0)[0,0])
            label = 1 if prob >= threshold else 0
            st.write({'sentence': text, 'probability': prob, 'label': label})
