import json
from tensorflow.keras.preprocessing.text import Tokenizer
from .base_tokenizer import BaseTokenizer

class KerasTokenizerWrapper(BaseTokenizer):
    def __init__(self, num_words=10000, oov_token='<OOV>'):
        self.num_words = num_words
        self.oov_token = oov_token
        self._tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)

    def fit(self, texts):
        self._tokenizer.fit_on_texts(texts)

    def texts_to_sequences(self, texts):
        return self._tokenizer.texts_to_sequences(texts)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'num_words': self.num_words,
                'oov_token': self.oov_token,
                'word_index': self._tokenizer.word_index
            }, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.num_words = data.get('num_words', self.num_words)
        self.oov_token = data.get('oov_token', self.oov_token)
        # rebuild tokenizer
        self._tokenizer = Tokenizer(num_words=self.num_words, oov_token=self.oov_token)
        self._tokenizer.word_index = data.get('word_index', {})
