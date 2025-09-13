from accelerate.utils import release_memory
from .shared import LoadUnloadMixin
from ..huggingface_model import HuggingFaceModel
from ..constants import ModelTag


class TextGenerationModel(LoadUnloadMixin, HuggingFaceModel):
    def __init__(self, id: str):
        super().__init__(id, "text-generation")
        self.tokenizer = None
        self.model = None

    

    def _prepare_inputs(self, text_inputs: str):
        return self.tokenizer(text_inputs, return_tensors="pt")

    def generate(
        self,
        text_inputs: str,
        **kwargs,
    ):
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Tokenizer and model must be loaded before generation.")
