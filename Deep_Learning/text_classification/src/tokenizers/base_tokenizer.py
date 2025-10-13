# ABC --> Abstract Base Class
# purpose is to establish a standard interface for any tokenizer 
# class used in the project
# ensures all specific tokenizer implementations behave the same way, 
# regardless of the underlying library (like Keras or Hugging Face)
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
