import logging

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

load_dotenv()

from models.rag.graph_rag import GraphRag

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

graph_rag = GraphRag("西游记", ChatDeepSeek(model="deepseek-chat", temperature=0))
# rag.clear_store()
graph_rag.load_to_store()

# def test_generation():
#    graph_rag = GraphRag("Elizabeth I", ChatDeepSeek(model="deepseek-chat", temperature=0))
#    response=graph_rag.retrieval_and_generation("Which house did Elizabeth I belong to?")
#    print(response)
