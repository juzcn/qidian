import os

from openai import OpenAI

from models.modelbase import ModelBase


class DouBaoCHAT(ModelBase):
    def __init__(self, model: str = "doubao-1-5-pro-32k-250115", base_url="https://ark.cn-beijing.volces.com/api/v3",temperature: float = 0):
        self.temperature = temperature
        self.model = model
        self.client = OpenAI(
            # 此为默认路径，您可根据业务所在地域进行配置
            base_url=base_url,
            api_key=os.getenv("ARK_API_KEY"),
        )

    def invoke(self, prompt, **call_args):
        completion = self.client.chat.completions.create(
            # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
            model=self.model,
            messages=prompt,
            temperature=self.temperature,
        )
        return completion

    def stream(self, prompt, **call_args):
        stream = self.client.chat.completions.create(
            # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
            model=self.model,
            messages=prompt,
            # 响应内容是否流式返回
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            yield chunk
