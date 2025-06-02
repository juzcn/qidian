import re

import requests
from markdownify import markdownify
from requests import RequestException
from smolagents import LiteLLMModel, load_tool, CodeAgent, WebSearchTool, tool, ToolCallingAgent, \
    ChatMessageStreamDelta, FinalAnswerStep

from models.modelbase import ModelBase


def smol_stream_get(token):
    if token is not None:
        if isinstance(token, ChatMessageStreamDelta) and token.content is not None:
            return token.content
    if isinstance(token, FinalAnswerStep) and token.output is not None:
        return token.output
    return ""

class CodeAgentClient(ModelBase):
    def __init__(self, model_id: str):
        model = LiteLLMModel(model_id=model_id)
        image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)
        self.agent = CodeAgent(tools=[WebSearchTool(), image_generation_tool], model=model, stream_outputs=True)
        # 解决中文问题
        self.agent.prompt_templates["system_prompt"] = self.agent.prompt_templates[
                                                           "system_prompt"] + "\nAnswer in original language of question!"

    def invoke(self, prompt, **call_args):
        #        response = self.agent.run(prompt)
        #        print("Smol Agent response",response)
        return self.agent.run(prompt)

    def stream(self, prompt, **call_args):
        for token in self.agent.run(prompt, stream=True):
            yield token


@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Convert the HTML content to Markdown
        markdown_content = markdownify(response.text).strip()

        # Remove multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        return markdown_content

    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


class MultiAgent(ModelBase):
    def __init__(self, model_id: str):
        model = LiteLLMModel(model_id=model_id)
        web_agent = ToolCallingAgent(
            tools=[WebSearchTool(), visit_webpage],
            model=model,
            max_steps=10,
            name="web_search_agent",
            description="Runs web searches for you.",
        )
        self.agent = CodeAgent(
            tools=[],
            model=model,
            managed_agents=[web_agent],
            additional_authorized_imports=["time", "numpy", "pandas"],
        )
        # 解决中文问题
        self.agent.prompt_templates["system_prompt"] = self.agent.prompt_templates[
                                                           "system_prompt"] + "\nAnswer in original language of question!"

    def invoke(self, prompt, **call_args):
        #        response = self.agent.run(prompt)
        #        print("Smol Agent response",response)
        return self.agent.run(prompt)

    def stream(self, prompt, **call_args):
        for token in self.agent.run(prompt, stream=True):
            yield token

