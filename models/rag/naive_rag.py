import logging
import os
from typing import Any

import bs4
from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_core.language_models import BaseChatModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.llm_templates import template_to_content
from models.rag.ragbase import RagBase
from utils.batch_processing import PersistentBatchProcessor
from utils.hg_embeddings import HG_Embeddings

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NaiveRag(RagBase):
    NAIVE_RAG = {
        "demo": {
            # Indexing
            "doc_url": "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "loader": WebBaseLoader,
            "loader_args": {
                'bs_kwargs': dict(
                    parse_only=bs4.SoupStrainer(
                        class_=("post-content", "post-title", "post-header")
                    )
                )
            },
            "collection_name": "example_collection",
            "embedding_function": HG_Embeddings("BAAI/bge-m3"),
            #            "persist_directory": "./chroma_db",
            "splitter": RecursiveCharacterTextSplitter,
            "splitter_args": {"chunk_size": 1000, "chunk_overlap": 200},
            # Retrieval and generation
            "template": "基础RAG"
        },
        "四大名著": {
            # Indexing
            "doc_url": "四大名著.md",
            "loader": TextLoader,
            "loader_args": {"encoding": "utf-8"},
            "collection_name": "Four_Famous_Books",
            "embedding_function": HG_Embeddings("BAAI/bge-large-zh-v1.5"),
            #           "persist_directory": "./chroma_db",
            "splitter": RecursiveCharacterTextSplitter,
            "splitter_args": {"separators": [
                "\n\n",
                "\n",
                " ",
                ".",
                ",",
                "\u200b",  # Zero-width space
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                "",
            ], "chunk_size": 1000, "chunk_overlap": 100},
            # Retrieval and generation
            "template": "基础RAG"
        }

    }

    def __init__(self, rag_name: str, llm: BaseChatModel):
        super().__init__(llm)
        self.rag = NaiveRag.NAIVE_RAG[rag_name]
        self.chroma_client = PersistentClient(os.getenv("CHROMA_PERSIST_DIR"))

    def clear_store(self):
        try:
            self.chroma_client.delete_collection(self.rag["collection_name"])
            logger.info(f"collection {self.rag["collection_name"]} successfully deleted")
        except:
            # collection not exist
            pass

    def load_to_store(self):
        vector_store = Chroma(
            client=self.chroma_client,
            collection_name=self.rag["collection_name"],
            embedding_function=self.rag["embedding_function"],
        )
        loader = self.rag["loader"](self.rag["doc_url"], **self.rag["loader_args"])
        docs = loader.load()
        logger.info("Document loaded")
        text_splitter = self.rag["splitter"](**self.rag["splitter_args"])
        all_splits = text_splitter.split_documents(docs)
        logger.info(f"Document splits:{len(all_splits)}")
        processor = PersistentBatchProcessor(
            data=all_splits,
            fn_proc=vector_store.add_documents,
            batch_size=10,
            checkpoint_freq=1,
        )
        processor.run()

    def retrieval_and_generation(self, question: str | list[dict[str, Any]]):
        # Define prompt for question-answering
        # prompt = hub.pull("rlm/rag-prompt")
        logger.debug(f"question: {question}")
        logger.debug(f"rag info : {self.rag}")
        if isinstance(question, list):
            question = self.rephrase_question(question)
        logger.debug(f"rephrased question: {question}")

        # Define state for application
        vector_store = Chroma(
            collection_name=self.rag["collection_name"],
            embedding_function=self.rag["embedding_function"],
            persist_directory=os.getenv("CHROMA_PERSIST_DIR"),  # Where to save data locally, remove if not necessary
        )

        # Define application steps
        retrieved_docs = vector_store.similarity_search(question)
        #    exit(-1)
        #    logger.debug(f"retrieved_docs: {retrieved_docs}")
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        logger.debug(f"retrieved_docs: {docs_content}")
        messages = template_to_content(self.rag['template'], {"question": question, "context": docs_content})
        response = self.llm.invoke(messages)
        return response.content
