import os

from langchain_deepseek import ChatDeepSeek

default_prompt = '''you are a helpful assistant'''


class GPT:
    def __init__(self, api_key, prompt=None, base_url=None, proxy=None):
        self.api_key = api_key
        self.base_url = base_url
        if proxy:
            os.environ['HTTP_PROXY'] = proxy
            os.environ['HTTPS_PROXY'] = proxy
        self.client = ChatDeepSeek(model="deepseek-chat", temperature=0)
        self.initialize(prompt)

    def initialize(self, prompt=None):
        """重置对话，清空历史消息。如果有提示，添加提示。
        
        Args:
            prompt (str): 提示信息，默认为 None。
            
        Returns:
            None
        """
        if prompt:
            self.messages = [{"role": "system", "content": prompt}]
        else:
            self.messages = [{"role": "system", "content": default_prompt}]

    def chat(self, prompt):
        """对话。

        Args:
            prompt (str): 用户输入。
            model (str): 模型，默认为 gpt-3.5-turbo。
        
        Returns:
            str: 模型回复。
        """
        self.messages.append({"role": "user", "content": prompt})
        response = self.client.invoke(
            self.messages,
        )
        reply = response.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply
