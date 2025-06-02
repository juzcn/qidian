import json

from importlib_resources import files
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.openai_dalle_image_generation import OpenAIDALLEImageGenerationTool
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.language_models import BaseChatModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent

from utils.decorators import sync_async_func, sync_async_gen


class ReactAgent:
    duckduckgo = DuckDuckGoSearchRun()
    tavily = TavilySearch(max_results=5, topic="general")
    arxiv = load_tools(
        ["arxiv"],
    )
    dalle_tool = OpenAIDALLEImageGenerationTool(api_wrapper=DallEAPIWrapper())

    #    image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)
    #    web_search = WebSearchTool()

    def __init__(self, server_config, llm: BaseChatModel):
        json_content = files('resources').joinpath(server_config).read_text(encoding='utf-8')
        self.servers_config = json.loads(json_content)
        self.llm = llm

    @sync_async_func
    async def invoke(self, prompt, **call_args):
        client = MultiServerMCPClient(self.servers_config)
        tools = await client.get_tools()
        tools.extend([ReactAgent.tavily, ReactAgent.dalle_tool])
        agent = create_react_agent(self.llm, tools)
        response = await agent.ainvoke({"messages": prompt})
        for message in response['messages']:
            message.pretty_print()
        return response

    @sync_async_gen
    async def stream(self, prompt, **call_args):
        client = MultiServerMCPClient(self.servers_config)
        tools = await client.get_tools()
        tools.extend([ReactAgent.tavily, ReactAgent.dalle_tool])
        agent = create_react_agent(self.llm, tools)
        async for stream_mode, chunk in agent.astream(
                {"messages": prompt},
                stream_mode=["messages", "values"]
        ):
            if stream_mode == "values":
                chunk['messages'][-1].pretty_print()
            else:
                token, metadata = chunk
                if metadata['langgraph_node'] != 'tools':
                    yield token
