import logging
from typing import List, Any

from langchain_community.document_loaders import WikipediaLoader, TextLoader
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import Neo4jVector
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j_graphrag.types import SearchType
from pydantic import BaseModel, Field

from models.rag.ragbase import RagBase
from utils.batch_processing import PersistentBatchProcessor
from utils.hg_embeddings import HG_Embeddings

# logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# logger.setLevel(logging.DEBUG)


def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()


def structured_output_chain(graph, llm):
    graph.query(
        "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

    # Extract entities from text
    class Entities(BaseModel):
        """Identifying information about entities."""

        names: List[str] = Field(
            ...,
            description="All the person, organization, or business entities that "
                        "appear in the text",
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are extracting organization and person entities from the text.",
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {question}",
            ),
        ]
    )

    entity_chain = prompt | llm.with_structured_output(Entities)
    return entity_chain


def structured_retriever(question: str, graph, llm):
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = structured_output_chain(graph, llm).invoke({"question": question})
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL (node, node) {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result


def retriever(question: str, graph, llm, vector_index):
    logger.debug(f"Search query: {question}")
    structured_data = structured_retriever(question, graph, llm)
    search_results = vector_index.similarity_search(question)
    logger.debug(f"Search results: {search_results}")
    unstructured_data = [el.page_content for el in search_results]
    final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ".join(unstructured_data)}
    """
    return final_data


class GraphRag(RagBase):
    GRAPH_RAG = {
        "Elizabeth I": {
            # Indexing
            "loader": WikipediaLoader,
            "loader_args": {"query": "Elizabeth I"},
            "splitter": RecursiveCharacterTextSplitter,
            "splitter_args": {"chunk_size": 1000, "chunk_overlap": 200},
            "embedding_function": HG_Embeddings("sentence-transformers/all-MiniLM-L6-v2"),
            "database": "demo"
        },
        "西游记": {
            # Indexing
            "loader": TextLoader,
            "loader_args": {"file_path": "./data/四大名著.md", "encoding": "utf-8"},
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
            "embedding_function": HG_Embeddings("BAAI/bge-large-zh-v1.5"),
            "database": "xyj"
        },
    }

    def __init__(self, rag_name: str, llm: BaseChatModel):
        super().__init__(llm)
        self.rag = GraphRag.GRAPH_RAG[rag_name]
        self.graph = Neo4jGraph(database=self.rag["database"], enhanced_schema=True)

    def clear_store(self):
        self.graph.query(f"MATCH (n) DETACH DELETE n")
        logger.info(f"Clear database: {self.rag['database']}")

    def load_to_store(self):
        loader = self.rag["loader"](**self.rag["loader_args"])
        docs = loader.load()
        logger.info("Document loaded")
        text_splitter = self.rag["splitter"](**self.rag["splitter_args"])
        all_splits = text_splitter.split_documents(docs)
        logger.info(f"Document splits:{len(all_splits)}")

        llm_transformer = LLMGraphTransformer(self.llm)

        def convert_and_add(batch):
            graph_documents = llm_transformer.convert_to_graph_documents(batch)
            self.graph.add_graph_documents(
                graph_documents=graph_documents,
                baseEntityLabel=True,
                include_source=True
            )

        processor = PersistentBatchProcessor(
            data=all_splits,
            fn_proc=convert_and_add,
            batch_size=1,
            checkpoint_freq=1
        )
        processor.run()

        Neo4jVector.from_existing_graph(
            self.rag["embedding_function"],
            search_type=SearchType.HYBRID,
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )

    def retrieval_and_generation(self, question: str | list[dict[str, Any]]):
        logger.debug(f"rag info : {self.rag}")
        if isinstance(question, list):
            question = self.rephrase_question(question)
        logger.debug(f"rephrased question: {question}")

        vector_index = Neo4jVector.from_existing_graph(
            self.rag["embedding_function"],
            search_type=SearchType.HYBRID,
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )

        template = """Answer the question based only on the following context:
        {context}
    
        Question: {question}
        Use natural language and be concise.
        Answer:"""
        context = retriever(question, self.graph, self.llm, vector_index)
        prompt = ChatPromptTemplate.from_template(template).invoke({"question": question, "context": context})

        return self.llm.invoke(prompt).content
