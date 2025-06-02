import os

from zhipuai import ZhipuAI


def glm_asr(file):
    """
    Name:
        语音识别服务
    Description:
        用输入的音频文件进行语音识别，返回识别结果
    Args:
        file: 用户输入的音频文件路径，例如"/path/to/audio.wav"
    Return：
        返回一个包含识别结果的dict数据
    """
    zhipu = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))  # 填写您自己的APIKey

    with open(file, "rb") as audio_file:
        transcriptResponse = zhipu.audio.transcriptions.create(
            model="glm-asr",
            file=audio_file,
            stream=False
        )
        return transcriptResponse.text
