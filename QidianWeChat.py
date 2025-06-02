import time

from dotenv import load_dotenv

# 读取相关环境变量
load_dotenv()
from wxauto import WeChat

from models.modelbase import model_invoke
from models.models_config import ModelsConfig

wx = WeChat()
# 指定监听目标
listen_list = [
    '张军',
    #    '李四',
    #    '工作群A',
    #    '工作群B'
]

friend_history = {}
for i in listen_list:
    wx.AddListenChat(who=i)  # 添加监听对象
    friend_history[i] = {
        "chatbot": [],
        'message_history': []
    }
# 持续监听消息，有消息则对接大模型进行回复
wait = 1  # 设置1秒查看一次是否有新消息
while True:
    msgs = wx.GetListenMessage()
    for chat in msgs:
        who = chat.who
        msg = msgs.get(chat)  # 获取消息内容
        for i in msg:
            if i.type == 'friend':
                friend_history[who]['chatbot'].append({"role": "user", "content": i.content})
                friend_history[who]['message_history'].append({"role": "user", "content": i.content})
                # Use Deepseek
                ModelsConfig["ReactAgent"]["model_args"]["class_args"]["model"] = "deepseek-chat"
                ModelsConfig["ReactAgent"]["model_args"]["class_args"]["temperature"] = 0.5

                reply = model_invoke(friend_history[who]['message_history'], ModelsConfig["ReactAgent"],
                                     ModelsConfig["ReactAgent"]["model_args"])
                friend_history[who]['chatbot'].append({"role": "assistant", "content": reply})
                friend_history[who]['message_history'].append({"role": "assistant", "content": reply})

                chat.SendMsg(reply)  # 回复
    time.sleep(wait)
