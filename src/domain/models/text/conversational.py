from transformers import TextIteratorStreamer
from .shared import LoadUnloadMixin, StreamMixin
from ..huggingface_model import HuggingFaceModel


class ConversationalModel(LoadUnloadMixin, StreamMixin, HuggingFaceModel):
    def __init__(self, id: str):
        super().__init__(id, "conversational")
        self.tokenizer = None
        self.model = None

    def _prepare_inputs(self, text_inputs: str):
        return self.tokenizer.apply_chat_template(
            text_inputs,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        )

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
            "input_ids":model_inputs,
            **config_kwargs,
            "max_length": 1024,
            **kwargs
        }

        if stream:
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            return self.stream_message(streamer, generation_kwargs)
        else:
            input_len = model_inputs.shape[1]
            generated_ids = self.model.generate(**generation_kwargs)
            new_tokens = generated_ids[:, input_len:]
            text = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
            return text
