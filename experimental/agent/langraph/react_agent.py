import asyncio
import json
from typing import Any

from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.language_models import BaseChatModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

load_dotenv(verbose=True)
from langchain_deepseek import ChatDeepSeek

with open('../mcp-servers.json', 'r', encoding='utf-8') as file:
    # Load the JSON data from the file
    server_config = json.load(file)

llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

questions = [
    "开车导航从北京到上海",
    "185除以3等于多少？",
    "What is the capital of France? Also, what is the population of that city?",
    "150除以3等于多少",
]


async def call_mcp_server(server_config: dict[str, Any], llm: BaseChatModel, prompt) -> Any:
    client = MultiServerMCPClient(server_config)
    tools = await client.get_tools()
    agent = create_react_agent(llm, tools)
    return await agent.ainvoke({"messages": prompt})


async def call_agent(llm: BaseChatModel, tools, prompt) -> Any:
    #    callbacks = CallbackManager([ConsoleCallbackHandler(), StdOutCallbackHandler(), UsageMetadataCallbackHandler()], ),
    agent = create_react_agent(llm, tools)
    return await agent.ainvoke({"messages": prompt})


async def get_mcp_tools(server_config: dict[str, Any]) -> Any:
    client = MultiServerMCPClient(server_config)
    return await client.get_tools()


async def main(question) -> Any:
    tools = await get_mcp_tools(server_config=server_config)
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    tools.append(wikipedia)
    return await call_agent(llm, tools, question)


# response = asyncio.run(main(questions[2]))
# for message in response["messages"]:
#     message.pretty_print()
if __name__ == '__main__':
    # non streaming mode
    response = asyncio.run(main(questions[0]))
    for message in response["messages"]:
        message.pretty_print()
