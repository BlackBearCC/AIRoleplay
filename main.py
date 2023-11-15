from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.agents.agent_types import AgentType

import LanguageModelSwitcher
import charact_selector
import prompt_manages
from langchain.prompts import ChatPromptTemplate
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

from role_tool import ChatTool,MemoryTool,ActionResponeTool,GameKnowledgeTool
from langchain.agents import initialize_agent
from LanguageModelSwitcher import LanguageModelSwitcher

# 创建 LanguageModelSwitcher 的实例
model = LanguageModelSwitcher("openai").model
# chain = final_prompt | model
# user_input = input()
# for chunk in chain.stream({"user": "大头", "char": "兔叽", "input":user_input}):
#      print(chunk)

tools = [ChatTool(),MemoryTool(),ActionResponeTool(),GameKnowledgeTool()]
agent = initialize_agent(tools,model, agent="zero-shot-react-description", verbose=True)
agent.run("游戏好玩吗")

#
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
