from typing import Iterator, Union
from ..domain.models.utils import list_local_models
from ..domain.models.text.text_generation import TextGenerationModel


class TextGenerationService:
    def __init__(self):
        self.assistant: TextGenerationModel = None
    
    def set_assistant(self, id: str):
        self.assistant = TextGenerationModel(id)
        self.assistant.load()
        return self.assistant
    
    def get_conversational_assistants_list(self):
        return list_local_models("text-generation")
    
    def send(self, content: str, stream=False, **kwargs) -> Union[str, Iterator[str]]:
        if not self.assistant:
            raise Exception("Assistant wasn't loaded correctly. This could be a caching problem")
        
        assistant_response = ""
        try:
            if stream:
                streamer = self.assistant.generate(
                    content, stream=True, **kwargs
                )
                return streamer
            else:
                assistant_response = self.assistant.generate(
                    content, stream=False, **kwargs
                )
                return assistant_response
        except Exception as e:
            raise e
