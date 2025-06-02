from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, List, Dict

from utils.decorators import function_logging


class ModelBase(ABC):
    @abstractmethod
    def invoke(self, prompt, **call_args):
        pass

    @abstractmethod
    def stream(self, prompt, **call_args):
        pass


@function_logging(handle_errors=False)
def model_invoke(prompt: str | List[Dict[str,Any]], model: dict[str, Any], model_args: dict[str, Any]) -> Any:
    response = model["model_class"](**model_args['class_args']).invoke(prompt, **model_args['call_args'])
    return model["invoke_get"](response)


@function_logging(handle_errors=False)
def model_stream(prompt: str | List[Dict[str,Any]], model: dict[str, Any], model_args: dict[str, Any]) -> Generator[Any, None, None]:
    for token in model["model_class"](**model_args['class_args']).stream(prompt, **model_args['call_args']):
        yield model["stream_get"](token)
