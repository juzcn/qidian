import logging

from dotenv import load_dotenv

from models.llm_models import ModelsConfig

load_dotenv()
from utils.hg_embeddings import HG_Embeddings
from models.modelbase import model_invoke, model_stream

# Set your logger to a high level (like ERROR)
# logging.disable(logging.CRITICAL)
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Print all active loggers
# print(logging.Logger.manager.loggerDict.keys())

# silenced_loggers = [
#    "baidubce.auth.bce_v1_signer",
#    "baidubce.auth",
#    "qianfan",
#    "baidubce",  # Often contains oauth.py
# ]
# for logger_name in silenced_loggers:
#    logger = logging.getLogger(logger_name)
#    logger.addHandler(logging.NullHandler())
#    logger.propagate = False


# qianfan_logger = logging.getLogger("qianfan")
# qianfan_logger.addHandler(logging.NullHandler())
# qianfan_logger.propagate = False

def test_hg_embeddings():
    logger.debug("Test HG Embeddings BEGIN ...")
    logger.debug(f"Supported models: {HG_Embeddings.Config.keys()}")
    for model in HG_Embeddings.Config.keys():
        logger.debug(f"model: {model}")
        embedding_function = HG_Embeddings(model)
        query = "今天天气怎么样？"
        embeddings = embedding_function.embed_query(query)
        logger.debug(f"embed_query: {query} -> {type(embeddings.shape)},{embeddings.shape}")
        docs = ["今天天气怎么样？", '今天天气很好']
        embeddings = embedding_function.embed_documents(docs)
        logger.debug(f"embed_documents: {docs} -> {type(embeddings.shape)},{embeddings.shape}")
        logger.info(f"Embeddings model {model} passed")
    logger.debug("Test HG Embeddings END")


def test_text_model(model_name, prompt_text):
    # Text to Text models
    # Three formats
    logger.debug(f"model: {model_name} prompt: {prompt_text}")
    prompt_message = [{"role": "user", "content": prompt_text}]
    prompt_multiple_messages = [{"role": "user", "content": [{'text': prompt_text, 'type': 'text'}]}]
    model_meta = ModelsConfig[model_name]
    try:
        response = model_invoke(prompt_text, model_meta, model_meta["model_args"])
        logger.debug(response)
    except:
        logger.info(f"llm model/app {model_name} do not support text message")
    response = model_invoke(prompt_message, model_meta, model_meta["model_args"])
    logger.debug(response)

    try:
        response = model_invoke(prompt_multiple_messages, model_meta, model_meta["model_args"])
        logger.debug(response)
    except:
        logger.info(f"llm model/app {model_name} do not support multimodal messages")

    logger.debug("Test for streaming")
    for token in model_stream(prompt_message, model_meta, model_meta["model_args"]):
        logger.debug(token)
    logger.info(f"llm model/app {model_name} passed")


def test_cogview(model_name, prompt_text):
    # Cogview model
    logger.debug(f"model: {model_name} prompt: {prompt_text}")
    model_meta = ModelsConfig[model_name]
    response = model_invoke(prompt_text, model_meta, model_meta["model_args"])
    logger.debug(response)
    logger.info(f"llm model/app {model_name} passed")


def test_glm4v(model_name, prompt_text, image_url):
    # Cogview model
    logger.debug(f"model: {model_name} prompt: {prompt_text}")
    prompt = [{"role": "user", "content": [{'text': prompt_text, 'type': 'text'},
                                           {'image_url': {'url': image_url}, 'type': 'image_url'}]}]
    model_meta = ModelsConfig[model_name]
    response = model_invoke(prompt, model_meta, model_meta["model_args"])
    logger.debug(response)
    logger.debug("Test for streaming")
    for token in model_stream(prompt, model_meta, model_meta["model_args"]):
        logger.debug(token)
    logger.info(f"llm model/app {model_name} passed")


def test_mcp_client(model_name, prompt_text, llm):
    # MCP Client
    logger.debug(f"model: {model_name} prompt: {prompt_text}")
    prompt_message = [{"role": "user", "content": prompt_text}]
    model_meta = ModelsConfig[model_name]
    model_meta['model_args']['class_args']['llm'] = llm

    model_args = model_meta["model_args"]
    # preprocessing model args

    #    llm = model_meta["model_args"]["class_args"]['llm']
    logger.debug(f"Use LLM model {llm} in class arguments")
    model_args["class_args"]["model"] = ModelsConfig[llm]["model_args"]["class_args"]["model"]
    model_args["class_args"]["temperature"] = ModelsConfig[llm]["model_args"]["class_args"]["temperature"]
    logger.debug(f"model args modified: {model_args}")
    try:
        response = model_invoke(prompt_message, model_meta, model_args)
        logger.debug(response)
    except NotImplementedError:
        logger.info(f"llm model/app {model_name} using {llm} ： Langchain NotImplementedError")
        return
    except AttributeError:
        logger.info(f"llm model/app {model_name} using {llm} ： Do not support langchain bind_tools")
        return

    logger.debug("Test for streaming")
    for token in model_stream(prompt_message, model_meta, model_args):
        logger.debug(token)
    logger.info(f"llm model/app {model_name} using {llm} passed")


def test_naive_rag(model_name, prompt_text, llm, name):
    # MCP Client
    logger.debug(f"model: {model_name} prompt: {prompt_text}")
    prompt_message = [{"role": "user", "content": prompt_text}]
    model_meta = ModelsConfig[model_name]
    model_meta['model_args']['class_args']['llm'] = llm
    model_meta['model_args']['class_args']['name'] = name

    model_args = model_meta["model_args"]
    # preprocessing model args

    #    llm = model_meta["model_args"]["class_args"]['llm']
    logger.debug(f"Use LLM model {llm} in class arguments")
    model_args["class_args"]["model"] = ModelsConfig[llm]["model_args"]["class_args"]["model"]
    model_args["class_args"]["temperature"] = ModelsConfig[llm]["model_args"]["class_args"]["temperature"]
    logger.debug(f"model args modified: ", model_args)
    response = model_invoke(prompt_message, model_meta, model_args)
    logger.debug(response)
    logger.info(f"llm model/app {model_name} using {llm} with {name} passed")


def test_graph_rag(model_name, prompt_text, llm, name):
    # MCP Client
    logger.debug(f"model: {model_name} prompt: {prompt_text}")
    prompt_message = [{"role": "user", "content": prompt_text}]
    model_meta = ModelsConfig[model_name]
    model_meta['model_args']['class_args']['llm'] = llm
    model_meta['model_args']['class_args']['name'] = name

    model_args = model_meta["model_args"]
    # preprocessing model args

    #    llm = model_meta["model_args"]["class_args"]['llm']
    logger.debug(f"Use LLM model {llm} in class arguments")
    model_args["class_args"]["model"] = ModelsConfig[llm]["model_args"]["class_args"]["model"]
    model_args["class_args"]["temperature"] = ModelsConfig[llm]["model_args"]["class_args"]["temperature"]
    logger.debug(f"model args modified: ", model_args)
    try:
        response = model_invoke(prompt_message, model_meta, model_args)
        logger.debug(response)
    except NotImplementedError:
        logger.info(f"llm model/app {model_name} using {llm} ： Langchain NotImplementedError")
        return
    except AttributeError:
        logger.info(f"llm model/app {model_name} using {llm} ： Langchain NotImplementedError")
        return
    logger.info(f"llm model/app {model_name} using {llm} with {name} passed")


# test_hg_embeddings()
# test_text_model("DeepSeekChat", "你好")
# test_text_model("MoonshotChat", "你好")
# test_text_model("DouBaoChat", "你好")
# test_text_model("QianfanChat", "你好")
# test_text_model("ZhipuChat", "你好")
# test_cogview("Cogview4", "请画一个猫")
# test_glm4v("GLM4V", "图中有什么", "https://cdn.deepseek.com/logo.png?x-image-process=image%2Fresize%2Cw_1920")
# test_mcp_client("LG-MCPClient", "北京天气怎么样", "DeepSeekChat")
# test_mcp_client("LG-MCPClient", "北京天气怎么样", "MoonshotChat")
# test_mcp_client("LG-MCPClient", "北京天气怎么样", "DouBaoChat")
# test_mcp_client("LG-MCPClient", "北京天气怎么样", "QianfanChat")
# test_mcp_client("LG-MCPClient", "北京天气怎么样", "ZhipuChat")
# test_naive_rag("NaiveRag", "What is Task Decomposition?", "DeepSeekChat", "demo")
# test_graph_rag("GraphRag", "Which house did Elizabeth I belong to?", "DeepSeekChat", "Elizabeth I")
