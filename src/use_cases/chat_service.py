from typing import Union, Iterator, List, Literal

from ..domain.models.text.conversational import ConversationalModel
from ..domain.models.utils import list_local_models


class ChatService:
    def __init__(self):
        self._messages: List[dict] = []
        self.assistant: ConversationalModel = None

    def set_assistant(self, id: str):
        self.assistant = ConversationalModel(id)
        self.assistant.load()
        return self.assistant

    def get_conversational_assistants_list(self):
        return list_local_models("conversational")

    def get_messages(self):
        return self._messages

    def set_messages(self, messages: List[dict]):
        self._messages = messages

    def set_system_message(self, content: str):
        if self._messages[0]["role"] == "system":
            self._messages[0]["content"] = content

    def append_message(
        self, role: Literal["system", "user", "assistant"], content: str
    ) -> dict:
        self._messages.append({"role": role, "content": content})

    def pop_message(self, index: int):
        if index > 0 and index < len(self.messages):
            msg = self._messages.pop(index)
            return msg
        return None

    def clear_messages(self) -> None:
        self._messages.clear()

    def send(self, content: str, stream=False, **kwargs) -> Union[str, Iterator[str]]:
        """Send a message to the conversational assistant and generate a response.

        Args:
            content (str): The content of the message to send to the assistant.
            stream (bool, optional): If `True`, returns a generator yielding response chunks as they are produced.
            Defaults to False.

        Returns:
            Union[str, Iterator[str]]: Depending on the `stream` flag, it either returns the full response as a string
            or an iterator that yields response chunks as they are generated.
        """
        self.append_message("user", content)
        assistant_response = ""
        try:
            if stream:
                streamer = self.assistant.generate(
                    self._messages, stream=True, **kwargs
                )
                return streamer
            else:
                assistant_response = self.assistant.generate(
                    self._messages, stream=False, **kwargs
                )
                return assistant_response
        except Exception as e:
            raise e
