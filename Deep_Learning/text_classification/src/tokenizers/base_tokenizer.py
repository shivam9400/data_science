from abc import ABC, abstractmethod

class BaseTokenizer(ABC):
    """Abstract tokenizer interface."""

    @abstractmethod
    def fit(self, texts):
        pass

    @abstractmethod
    def texts_to_sequences(self, texts):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass
