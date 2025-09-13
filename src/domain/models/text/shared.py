from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from torch import cuda
from accelerate.utils import release_memory


class LoadUnloadMixin:
    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, padding_side="left")

        if cuda.is_available():
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.path, device_map="auto", quantization_config=quantization_config
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.path)

        return self.tokenizer, self.model

    def unload(self):
        release_memory(self.tokenizer, self.model)


class StreamMixin:
    def stream_message(self, streamer, generation_kwargs) -> TextIteratorStreamer:
        generation_kwargs["streamer"] = streamer
        thread = Thread(
            target=self.model.generate, kwargs=generation_kwargs, daemon=True
        )
        thread.start()
        return streamer
