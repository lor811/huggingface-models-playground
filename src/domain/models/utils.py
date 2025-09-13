import os
from typing import List
from .constants import ModelTag, MODELS_DIR


def list_local_models(tag: ModelTag) -> List[str]:
    try:
        path = os.path.join(MODELS_DIR, tag)
        return [
            name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
        ]
    except FileNotFoundError:
        return []
