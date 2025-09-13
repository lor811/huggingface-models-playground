import os
from typing import Literal


MODELS_DIR = os.path.join(".", ".models")
ModelTag = Literal[
    "conversational",
    "text-generation",
]
