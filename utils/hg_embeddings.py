import logging
from typing import List

# load env before loading huggingface model
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from numpy import ndarray

load_dotenv()
from sentence_transformers import SentenceTransformer

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class HG_Embeddings(Embeddings):
    Config = {
        "BAAI/bge-large-zh-v1.5": {
            "model_func": SentenceTransformer,
            "model_args": {
                "trust_remote_code": True
            }
        },
        "TencentBAC/Conan-embedding-v2": {
            "model_func": SentenceTransformer,
            "model_args": {
            }
        },
        "BAAI/bge-m3": {
            "model_func": SentenceTransformer,
            "model_args": {
                "trust_remote_code": True
            }
        },
        "sentence-transformers/all-MiniLM-L6-v2": {
            "model_func": SentenceTransformer,
            "model_args": {
                "trust_remote_code": True
            }
        },
    }

    # Using HuggingFace
    def __init__(self, model_name: str):
        self.model = HG_Embeddings.Config[model_name]['model_func'](model_name,
                                                                    **HG_Embeddings.Config[
                                                                                      model_name]['model_args'])

    def embed_documents(self, texts: List[str]) -> ndarray:
        """Embed search docs."""
        return self.model.encode(texts)

    def embed_query(self, text: str) -> ndarray:
        """Embed query text."""
        return self.model.encode([text])[0]

# JinaV2BaseZhEmbeddings = lambda **kwargs: HG_Embeddings(AutoModel.from_pretrained,"jinaai/jina-embeddings-v2-base-zh",trust_remote_code=True,**kwargs)
# Text2VecBaseChineseEmbeddings = lambda **kwargs: HG_Embeddings(SentenceModel,"shibing624/text2vec-base-chinese")
# AllMiniLMEmbeddings = lambda **kwargs: HG_Embeddings(SentenceTransformer,"sentence-transformers/all-MiniLM-L6-v2",trust_remote_code=True,**kwargs)
# JinaV3Embeddings = lambda **kwargs: HG_Embeddings(AutoModel.from_pretrained,"jinaai/jina-embeddings-v3",trust_remote_code=True,**kwargs)

# AlibabaQwenEmbeddings = lambda **kwargs: HG_Embeddings(SentenceTransformer,"Alibaba-NLP/gte-Qwen2-7B-instruct",trust_remote_code=True,**kwargs)
# LingMistralEmbeddings = lambda **kwargs: HG_Embeddings(SentenceTransformer,"Linq-AI-Research/Linq-Embed-Mistral",trust_remote_code=True,**kwargs)

# BAAIBgeLargeZh = lambda **kwargs: HG_Embeddings(SentenceTransformer,"BAAI/bge-large-zh-v1.5",trust_remote_code=True,**kwargs)
