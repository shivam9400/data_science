import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'src' package can be imported when this file is run
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
from pathlib import Path

# Try importing TensorFlow components and keep any import error for a helpful UI message
tf_import_error = None
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception as e:
    load_model = None
    pad_sequences = None
    tf_import_error = e

from src.tokenizers.keras_tokenizer import KerasTokenizerWrapper
from src.utils.io import load_json

st.set_page_config(page_title='Text Classifier', layout='centered')
st.title('Modular Text Classification')

artifact_dir = st.sidebar.text_input('Artifacts directory', value='artifacts')

if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'tokenizer' not in st.session_state:
    st.session_state['tokenizer'] = None
if 'config' not in st.session_state:
    st.session_state['config'] = None

if st.sidebar.button('Load model'):
    if tf_import_error is not None:
        st.error(f"TensorFlow import failed: {tf_import_error}.\nInstall TensorFlow in your environment and restart the app.")
    else:
        model_path = Path(artifact_dir) / 'model.h5'
        tokenizer_path = Path(artifact_dir) / 'tokenizer.json'
        config_path = Path(artifact_dir) / 'config.json'
        if not model_path.exists():
            st.error('Model not found in artifacts. Train and save a model first.')
        else:
            st.session_state['model'] = load_model(model_path)
            st.session_state['config'] = load_json(config_path)
            st.session_state['tokenizer'] = KerasTokenizerWrapper(...)
            st.session_state['tokenizer'].load(tokenizer_path)
            st.success('Model loaded')

# Initialize session state for the result
if 'classification_result' not in st.session_state:
    st.session_state['classification_result'] = None

# Check if the model is loaded in the session state
if st.session_state['model'] is not None:
    text = st.text_area('Enter text to classify', key='input_text')
    threshold = st.slider('Threshold', 0.0, 1.0, 0.5)

    if st.button('Classify'):
        # 2. Retrieve the text directly from session state using the key
        text_input = st.session_state['input_text']

        # Retrieve objects from session state for use
        model = st.session_state['model']
        tokenizer = st.session_state['tokenizer']
        config = st.session_state['config']
        
        # --- Classification logic using the retrieved objects ---
        seq = tokenizer.texts_to_sequences([text])
        pad = pad_sequences(seq, maxlen=config.get('max_len', 20), padding='post')
        prob = float(model.predict(pad, verbose=0)[0,0])
        label = 1 if prob >= threshold else 0
        class_label = "positive" if label==1 else "negative"
        st.write({'sentence': text, 
                  'probability': prob, 
                  'label': label,
                  'sentiment':class_label})
else:
    st.info("Please load a model using the button in the sidebar.")