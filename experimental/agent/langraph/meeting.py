# Meeting LLM app
# One coordinator role, n participants, each have a llm behind
# Everyone has a fictive name associated, optionally has a role assignment
# App input: topic and general requirement
# Meeting Logic: Mode first decide first say
#       1. Coordinator explain topic with general requirement, and announce the beginning
#       2. Each participant make a judgement, have "something to say" or "nonthing to say"
#       3. The fist who made a judgement "something to say" can say, others aborted
#       4. When no participant have something to say, Coordinator conclude and announce meeting closed


from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi, MoonshotChat, ChatZhipuAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, SerpAPIWrapper
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_deepseek import ChatDeepSeek
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent

load_dotenv()

tavily = TavilySearch(
    max_results=5,
    topic="general",
)

deepseek_chat = ChatDeepSeek(model="deepseek-chat", temperature=0)
tongyi_chat = ChatTongyi(model="qwen-plus", top_p=0.5)
moonshot_chat = MoonshotChat(model="moonshot-v1-8k", temperature=0.5)
zhipu_chat = ChatZhipuAI(model="glm-4-plus", temperature=0.2)
tools = [tavily]
search = SerpAPIWrapper()
custom_tool = Tool(
    name="web search",
    description="Search the web for information",
    func=search.run,
)
topic = "中美之间的硬脱钩会不会发生",
mode = "轮流"  # “自由” or “轮流” or "举手" or "点名"
finish_mode = "到时间"  # “到时间” or "主持人根据内容" or "中断"
time = 5
participant_template = """你是{participant}, 你正在参加一个关于“{topic}”的研讨会,现在轮到您发言。\n以下是参会人员的发言情况：
{talk_history}
你可以使用的工具有{tools}。
请按研讨会主持人{coordinator}要求发言。
"""
coordinator_template = """我是主持人{coordinator}.今天讨论的话题是“{topic}”。参加今天研讨会的有{participants}。请参会人员按{mode}发言。你的发言应尽量简洁。不要与其他发言重复。可以直接回答“我没有新的内容补充”，或者‘我同意某某发言’。"
"""
talk_template = """{name}：{content}\n"""
coordinator_template_judge = """你是主持人{coordinator}.根据大家的发言{talk_history}，如果觉得其他参会人意见基本一致，可以简单做一个总结，结束会议，否则继续新一轮发言。
"""


class Coordinator:
    def __init__(self, name, topic, participants, mode=mode, finish_mode=finish_mode, time=time,
                 llm=deepseek_chat, tools=tools):
        self.name = name
        self.topic = topic
        self.participants = participants
        self.mode = mode
        self.finish_mode = finish_mode
        self.time = time
        self.llm = llm
        self.tools = tools
        self.talk_history = []
        self.agent = create_react_agent(llm, tools)

    def talk(self, content):
        self.talk_history.append({"name": self.name, "content": content})

    def start_meeting(self):
        coordinator_content = PromptTemplate.from_template(coordinator_template).invoke(
            {"coordinator": self.name, "topic": self.topic,
             "participants": [participant.name for participant in self.participants],
             "mode": self.mode
             }
        ).text
        self.talk_history.append({"name": self.name, "content": coordinator_content})
        for participant in self.participants:
            participant.required_to_talk(self.topic, self.talk_history, self.name)
        talk_history_text = ""
        for talk in self.talk_history:
            #            print(PromptTemplate.from_template(talk_template).invoke(talk).text)
            talk_history_text = talk_history_text + PromptTemplate.from_template(talk_template).invoke(talk).text
        prompt = PromptTemplate.from_template(coordinator_template_judge).invoke(
            {"coordinator": self.name,
             "talk_history": talk_history_text,
             }).text
        response = self.agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        #        for message in response["messages"]:
        #            message.pretty_print()
        self.talk_history.append({"name": self.name, "content": response["messages"][-1].content})
        print(self.talk_history)


class Participant:
    def __init__(self, name, llm: BaseChatModel = zhipu_chat, tools=[]):
        self.name = name
        self.llm = llm
        self.tools = tools
        self.agent = create_react_agent(llm, tools)

    def required_to_talk(self, topic, talk_history, coordinator):
        talk_history_text = ""
        for talk in talk_history:
            #            print(PromptTemplate.from_template(talk_template).invoke(talk).text)
            talk_history_text = talk_history_text + PromptTemplate.from_template(talk_template).invoke(talk).text
        prompt = PromptTemplate.from_template(participant_template).invoke(
            {"participant": self.name, "topic": topic,
             "talk_history": talk_history_text,
             "coordinator": coordinator,
             "tools": [tool.name for tool in self.tools]}).text
        response = self.agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        #        for message in response["messages"]:
        #            message.pretty_print()
        talk_history.append({"name": self.name, "content": response["messages"][-1].content})
        print(talk_history)

    def announce_end_received(self):
        pass

    def new_talk_received(self):
        pass


wang = Participant("王先生", deepseek_chat, tools=[tavily])
liu = Participant("刘小姐", zhipu_chat, tools=[WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())])
zhang = Participant("张先生", tongyi_chat, tools=[custom_tool])
coordinator = Coordinator("马云", topic, [wang, liu, zhang], "轮流", "到时间", 5)
coordinator.start_meeting()
