# HuggingFace connection
from smolagents import CodeAgent, WebSearchTool, InferenceClientModel, LiteLLMModel
from dotenv import load_dotenv
load_dotenv()
model = LiteLLMModel(model_id="deepseek/deepseek-chat")
# model = InferenceClientModel()
agent = CodeAgent(tools=[WebSearchTool()], model=model, stream_outputs=True)
agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
