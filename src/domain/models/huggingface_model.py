from abc import ABC, abstractmethod
from .constants import ModelTag, MODELS_DIR
import os


class HuggingFaceModel(ABC):
    def __init__(self, id: str, tag: ModelTag):
        self.id = id
        self.tag = tag
        self.path = os.path.join(MODELS_DIR, tag, id)

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def unload(self):
        pass

    @abstractmethod
    def generate(self, text_inputs: str, **kwargs):
        pass

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, HuggingFaceModel) and self.id == other.id
