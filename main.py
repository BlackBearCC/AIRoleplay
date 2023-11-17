from typing import Union, List

from langchain.chains import LLMChain
from langchain.chains.question_answering.map_rerank_prompt import output_parser
from langchain.memory import ConversationBufferMemory
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool, BaseTool
from langchain.vectorstores.chroma import Chroma
from langchain.agents.agent_types import AgentType

import LanguageModelSwitcher
import charact_selector
import prompt_manages
from langchain.prompts import ChatPromptTemplate, BaseChatPromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.embeddings import HuggingFaceBgeEmbeddings

from langchain.document_loaders import JSONLoader
import json
from pathlib import Path
from pprint import pprint

from langchain.document_loaders.csv_loader import CSVLoader


import os

from langchain.chat_models import ChatOpenAI


# llm = OpenAI(temperature=0)

# loader = CSVLoader(file_path='D:/AIAssets/ProjectAI/AIRoleplay/tangshiye_test_output_dialogue.csv')
# data = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# texts = text_splitter.split_documents(data)
# model_name = "BAAI/bge-large-en-v1.5"
# encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
#
# embedding_model = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs=encode_kwargs
# )
# query = "你是谁"
# vectordb = Chroma.from_documents(documents=texts,embedding=embedding_model)
# docs = vectordb.search(query=query,search_type="similarity", k=5)
# print(docs[0].page_content)
# prompt = prompt_manages.rolePlay()+tuji+prompt_manages.charactorStyle()+prompt_manages.plotDevelopment()+prompt_manages.prepare_corpus()
# final_prompt = ChatPromptTemplate.from_template(prompt)
# print(prompt)
from role_tool import ChatTool,MemoryTool,ActionResponeTool,GameKnowledgeTool
from langchain.agents import initialize_agent, ZeroShotAgent, AgentOutputParser, LLMSingleActionAgent, AgentExecutor
from LanguageModelSwitcher import LanguageModelSwitcher
import re
# 创建 LanguageModelSwitcher 的实例

model = LanguageModelSwitcher("text_gen").model





tools = [ChatTool()
]



# chain = final_prompt | model
# user_input = input()
# for chunk in chain.stream({"user": "大头", "char": "兔叽", "input":user_input}):
#      print(chunk)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# tools = [ChatTool(),MemoryTool(),ActionResponeTool(),GameKnowledgeTool()]
# agent = initialize_agent(tools,model, AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True,memory=memory)
# agent.run("游戏好玩吗")
template = """
进入角色扮演模式，代入{char}这个角色，在{char}和{user}之间展开对话。只回答{char}的话，你永远不会决定和描述{user}的台词和行为，而是以互联网角色扮演的风格来写作，例如不使用引号，用斜体字来表示{char}的动作。
     。要积极主动地推动剧情和对话的发展，利用{char}的特点和风格来与{user}互动,使角色扮演更有趣和引人入胜。请保持情感和角色特点的一致性。至少写一段，最多写四段，如果生成了多条回应，请选择并重复只包含{char}的一部分。要始终保持角色扮演的真实性。重复的内容可以省略。
     charactorStyle\n" + "{char} 可以主动提出问题或展示好奇，例如：“哇，为什么你长得跟我不一样呀？”,“哇！这是什么怪东西？！”,“嘿嘿嘿嘿，脸长长的会变成大蠢驴哦~”,“呜哇！肚子要饿扁了啦！”
“啊啊啊啊，我的木马骑士要吃成大肥猪头了！",“吧唧吧唧~胡萝卜世界第一无敌美味"
    {char} 可以回应{user}的提问，或者提供一些情景或目标，例如：*“这个森林里据说有超级大的胡萝卜，我们可以试着找到它。”*
    {char} 可以展示好奇心和兴趣，或者做出有趣的动作，例如：*{char}摇动着她的大耳朵，好奇地张望四周，看是否有什么迹象。*
    {char} 可以展示自己的情绪，例如：*{char}的小脸蛋红扑扑的，她的眼睛里闪着好奇的光芒。*
    {char}坐在地上，揉了揉眼睛，睡眼惺忪的打了个大大的哈欠，胖乎乎的小肉手在地上一通乱摸，仿佛还不相信自己已经结结实实的坐在地板上了。
    {user}:{input}
    {char}:"
Complete the objective as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: 输入原问题让[{tool_names}]输出答案
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: Okay, so what is the response to my last comment? If using information obtained from the tools you must mention it explicitly without mentioning the tool names - I have forgotten all TOOL RESPONSES! Remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else - even if you just want to respond to the user. Do NOT respond with anything except a JSON snippet no matter what! 

These were previous tasks you completed:



Begin!

Question: {input}
{agent_scratchpad}"""

class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[BaseTool]

    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        kwargs["char"] = "兔叽"
        kwargs["user"] = "大头"
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)
class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()
llm  = model
# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
agent_executor.run("你是谁.")


# agent.run("你的兴趣")


# retriever = vectordb.as_retriever(search_kwargs={"k": 5})

#
# # data = json.loads(Path(file_path).read_text())
# print(data)


# print(llm.predict("nihao!"))
# minimax = llm_selector.initialize_minimax()




# print(prompt)
#
# # file_path='tangshiye_test_output_dialogue.jsonl'
# # loader = CSVLoader(file_path=r'tangshiye_test_output_dialogue.csv')
# # data = loader.load()
# # data = json.loads(Path(file_path).read_text())
# # print(data)
# # chain = final_prompt | minimax
# chain = final_prompt | qianfan

#
