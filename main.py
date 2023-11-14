from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.agents.agent_types import AgentType

import llm_selector
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

from langchain.llms import OpenAI
import os
os.environ["OPENAI_API_KEY"] = "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"
from langchain.chat_models import ChatOpenAI

llm = OpenAI(temperature=0)

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

from role_tool import KnowledgeTool
from role_tool import ActTool
from langchain.agents import initialize_agent

tools = [KnowledgeTool(),ActTool()]
agent = initialize_agent(tools,llm,agent = "zero-shot-react-description",verbose=True)

agent.run("你的兴趣")


# retriever = vectordb.as_retriever(search_kwargs={"k": 5})

#
# # data = json.loads(Path(file_path).read_text())
# print(data)


# print(llm.predict("nihao!"))
# minimax = llm_selector.initialize_minimax()
# qianfan = llm_selector.initialize_qianfan()
# # model = OpenAI()
# #
# #
# tuji = charact_selector.initialize_tuji()

# prompt = prompt_manages.rolePlay()+tuji+prompt_manages.charactorStyle()+prompt_manages.plotDevelopment()+prompt_manages.prepare_corpus()
# final_prompt = ChatPromptTemplate.from_template(prompt)
# print(prompt)
#
# # file_path='tangshiye_test_output_dialogue.jsonl'
# # loader = CSVLoader(file_path=r'tangshiye_test_output_dialogue.csv')
# # data = loader.load()
# # data = json.loads(Path(file_path).read_text())
# # print(data)
# # chain = final_prompt | minimax
# chain = final_prompt | qianfan
# chain = final_prompt | model
#
# #
# #
# user_input = input()
# #

# # len(embedding)
# # print(embedding)
#
#
# for chunk in chain.stream({"user": "大头", "char": "兔叽", "input": "None"}):
#      print(chunk)
#
