from transformers import TextIteratorStreamer
from .shared import LoadUnloadMixin, StreamMixin
from ..huggingface_model import HuggingFaceModel


class TextGenerationModel(LoadUnloadMixin, StreamMixin, HuggingFaceModel):
    def __init__(self, id: str):
        super().__init__(id, "text-generation")
        self.tokenizer = None
        self.model = None

    def _prepare_inputs(self, text_inputs: str):
        return self.tokenizer(text_inputs, return_tensors="pt")

    def generate(
        self,
        text_inputs: str,
        stream=False,
        **kwargs,
    ):
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Tokenizer and model must be loaded before generation.")

        model_inputs = self._prepare_inputs(text_inputs).to(self.model.device)
        config_kwargs = self.model.generation_config.to_dict()
        generation_kwargs = {
            **model_inputs,
            **config_kwargs,
            **kwargs,
        }

        if stream:
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=False,
                skip_special_tokens=True,
            )
            return self.stream_message(streamer, generation_kwargs)
        else:
            generated_ids = self.model.generate(**generation_kwargs)
            text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text
