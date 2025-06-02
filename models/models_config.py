import json
import logging
import os

from importlib_resources import files

from models.agent.smol_codeagent import CodeAgentClient, MultiAgent, smol_stream_get
from models.mcp.react_agent import ReactAgent

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI, MoonshotChat, QianfanChatEndpoint, ChatTongyi
from langchain_deepseek import ChatDeepSeek
from zhipuai import ZhipuAI

from models.llm.doubao import DouBaoCHAT
from models.modelbase import ModelBase
from models.rag.graph_rag import GraphRag
from models.rag.naive_rag import NaiveRag

load_dotenv()
zhipu = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))  # 填写您自己的APIKey


class ReactAgentClient(ModelBase):
    def __init__(self, **class_args):
        llm = ModelsConfig[class_args['llm']]
        class_args_copy = class_args.copy()
        del class_args_copy['servers_config']
        del class_args_copy['llm']
        llm = llm['model_class'](**class_args_copy)
        self.client = ReactAgent(class_args['servers_config'], llm)

    def invoke(self, prompt, **call_args):
        return self.client.invoke(prompt)

    def stream(self, prompt, **call_args):
        for token in self.client.stream(prompt, **call_args):
            yield token


class NaiveRagClient(ModelBase):
    def __init__(self, **class_args):
        llm = ModelsConfig[class_args['llm']]
        class_args_copy = class_args.copy()
        del class_args_copy['name']
        del class_args_copy['llm']
        llm = llm['model_class'](**class_args_copy)
        self.client = NaiveRag(class_args['name'], llm)

    def invoke(self, prompt, **call_args):
        return self.client.retrieval_and_generation(prompt)

    def stream(self, prompt, **call_args):
        pass


class GraphRagClient(ModelBase):
    def __init__(self, **class_args):
        llm = ModelsConfig[class_args['llm']]
        class_args_copy = class_args.copy()
        del class_args_copy['name']
        del class_args_copy['llm']
        llm = llm['model_class'](**class_args_copy)
        self.client = GraphRag(class_args['name'], llm)

    def invoke(self, prompt, **call_args):
        return self.client.retrieval_and_generation(prompt)

    def stream(self, prompt, **call_args):
        pass


class COGVIEW_4(ZhipuAI):
    def __init__(self, **class_args):
        super().__init__(api_key=os.getenv("ZHIPUAI_API_KEY"), **class_args)

    def invoke(self, prompt, **call_args):
        response = self.images.generations(prompt=prompt, **call_args)
        return response

    def stream(self, prompt, model_args):
        raise NotImplementedError()


# Load models_config


ModelID2Class = {
    "DeepSeekChat": ChatDeepSeek,
    "ChatTongyi": ChatTongyi,
    "MoonshotChat": MoonshotChat,
    "DouBaoChat": DouBaoCHAT,
    "QianfanChat": QianfanChatEndpoint,
    "ZhipuChat": ChatZhipuAI,
    "ReactAgent": ReactAgentClient,
    "Cogview4": COGVIEW_4,
    "GLM4V": ChatZhipuAI,
    "NaiveRag": NaiveRagClient,
    "GraphRag": GraphRagClient,
    "CodeAgent": CodeAgentClient,
    "MultiAgent": MultiAgent
}

OutputParsers = {
    "content": lambda response: response.content,
    "choices[0].message.content": lambda response: response.choices[0].message.content,
    "choices[0].delta.content": lambda token: token.choices[0].delta.content,
    "['messages'][-1].content": lambda response: response['messages'][-1].content,
    "data[0].url": lambda response: response.data[0].url,
    "": lambda response: response,
    "smol_stream_get": smol_stream_get,
    None: None,
}
json_content = files('resources').joinpath("models_config.json").read_text(encoding='utf-8')
ModelsConfig = json.loads(json_content)
for key in ModelsConfig.keys():
    ModelsConfig[key]['model_class'] = ModelID2Class[key]
    ModelsConfig[key]['invoke_get'] = OutputParsers[ModelsConfig[key]['invoke_get']]
    ModelsConfig[key]['stream_get'] = OutputParsers[ModelsConfig[key]['stream_get']]
