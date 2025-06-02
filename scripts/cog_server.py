# llm_server.py
import logging
import os
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from zhipuai import ZhipuAI
"""
  "cog": {
    "command": "python",
    "args": [
      "scripts/cog_server.py"
    ],
    "transport": "stdio"
  },
"""
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
load_dotenv()
zhipu = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))  # 填写您自己的APIKey

mcp = FastMCP("Cog")


@mcp.tool()
def cogview(prompt: str) -> dict[str, Any]:
    """
    Name:
        文生图服务

    Description:
        用输入的图示词生成图片。

    Args:
        prompt: 用户输入的提示词文本，例如”生成一只可爱的猫咪”。

    Return：
        返回一个包含图片url的dict数据，例子如下
        {
            "created": 1703485556,
            "data": [
                {
                "url": "https://......"
                }
            ]
        }

    """
    print("#######", prompt)
    response = zhipu.images.generations(
        model="cogView-4-250304",  # 填写需要调用的模型编码
        prompt=prompt,
        size="1440x720"
    )
    logger.info(response)
    return response.data[0].url


@mcp.tool()
def cogvideox_image(prompt, image_url):
    """
    Name:
        图生视频服务

    Description:
        用输入的图和提示词创建建生成视频的任务单，稍候用户可以用任务单中的id检查任务状态，如果已完成，泽可以得到视频url

    Args:
        prompt: 用户输入的提示词文本，例如"让图片动起来。"
        image_url: 用户输入的图片文件路径URL，例如"/path/to/image.jpg"

    Return：
        返回一个生成视频的任务单，例子如下：
            id='8868902201637896192' request_id='654321' model='cogvideox-2' task_status='PROCESSING'

    """
    logger.info("cogvideox_image:" + prompt + " " + image_url)
    response = zhipu.videos.generations(
        model="cogvideox-2",  # 使用的视频生成模型
        #        image_url=file_to_base64(image_path),  # 提供的图片URL地址或者 Base64 编码
        image_url=image_url,  # 提供的图片URL地址或者 Base64 编码
        prompt=prompt,
        quality="quality",  # 输出模式，"quality"为质量优先，"speed"为速度优先
        with_audio=True,
        size="1920x1080",  # 视频分辨率，支持最高4K（如: "3840x2160"）
        fps=30,  # 帧率，可选为30或60
    )
    logger.info("cogvideox_image:")
    return str(response)


@mcp.tool()
def cogvideox_text(prompt):
    """
    Name:
        文生视频服务

    Description:
        用输入的图示词创建生成视频的任务单，稍候用户可以用任务单中的id检查任务状态，如果已完成，泽可以得到视频url

    Args:
        prompt: 用户输入的提示词文本，例如"比得兔开小汽车，游走在马路上，脸上的表情充满开心喜悦。"

    Return：
        返回一个生成视频的任务单，例子如下：
            id='8868902201637896192' request_id='654321' model='cogvideox-2' task_status='PROCESSING'

    """
    logger.info("cogvideox_text:" + prompt)
    response = zhipu.videos.generations(
        model="cogvideox-2",
        prompt=prompt,
        quality="quality",  # 输出模式，"quality"为质量优先，"speed"为速度优先
        with_audio=True,
        size="1920x1080",  # 视频分辨率，支持最高4K（如: "3840x2160"）
        fps=30,  # 帧率，可选为30或60
    )
    logger.info("cogvideox_text:")
    #    print(str(response))
    return str(response)


@mcp.tool()
def cogvideox_result(id: str):
    """
    Name:
        查询文生视频任务的结果

    Description:
        用输入的任务单id，查询文生视频任务的结果

    Args:
        id: 文生视频任务单的id

    Return：
        返回一个生成视频的任务单或视频

    """
    response = zhipu.videos.retrieve_videos_result(
        id=id
    )
    logger.info(response)
    return str(response)


if __name__ == "__main__":
    mcp.run(transport="stdio")
