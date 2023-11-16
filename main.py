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

model = LanguageModelSwitcher("minimax").model





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
template = """Complete the objective as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

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
