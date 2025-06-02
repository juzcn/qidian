import logging

from dotenv import load_dotenv

load_dotenv()
from langchain_deepseek import ChatDeepSeek

from models.rag.naive_rag import NaiveRag

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

rag = NaiveRag("demo", ChatDeepSeek(model="deepseek-chat", temperature=0))
rag.clear_store()
rag.load_to_store()
response = rag.retrieval_and_generation("What is Task Decomposition?")
print(response)

rag = NaiveRag("四大名著", ChatDeepSeek(model="deepseek-chat", temperature=0))
rag.clear_store()
rag.load_to_store()
response = rag.retrieval_and_generation("刘备的父亲是谁?")
print(response)
