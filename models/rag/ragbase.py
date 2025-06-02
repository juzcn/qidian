from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate


class RagBase(ABC):
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    @abstractmethod
    def load_to_store(self):
        pass

    @abstractmethod
    def retrieval_and_generation(self, prompt: str | list[dict[str, Any]]):
        pass

    @abstractmethod
    def clear_store(self):
        pass

    def rephrase_question(self, chat_history: list[dict[str, Any]]) -> str:
        question = chat_history[-1]['content']
        _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
        in its original language.
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""  # noqa: E501
        chat_history = PromptTemplate.from_template(_template).invoke(
            {"chat_history": chat_history[:-1], "question": question})
        question = self.llm.invoke(chat_history).content
        return question
