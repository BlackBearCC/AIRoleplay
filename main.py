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
llm = OpenAI()


print(llm.predict("nihao!"))
# minimax = llm_selector.initialize_minimax()
# qianfan = llm_selector.initialize_qianfan()
# model = OpenAI()
#
#
# tuji = charact_selector.initialize_tuji()
#
# prompt = prompt_manages.rolePlay()+tuji+prompt_manages.charactorStyle()+prompt_manages.plotDevelopment()
# final_prompt = ChatPromptTemplate.from_template(prompt)
#
# # file_path='tangshiye_test_output_dialogue.jsonl'
# # loader = CSVLoader(file_path=r'tangshiye_test_output_dialogue.csv')
# # data = loader.load()
# # data = json.loads(Path(file_path).read_text())
# # print(data)
# # chain = final_prompt | minimax
# # chain = final_prompt | qianfan
# chain = final_prompt | model
#
# #
# #
# user_input = input()
# #
# # model_name = "BAAI/bge-large-en-v1.5"
# # model_kwargs = {'device': 'cuda'}
# # encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
# # hf = HuggingFaceBgeEmbeddings(
# #     model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
# # )
# # embedding = hf.embed_query("hi this is harrison")
# # len(embedding)
# # print(embedding)
#
#
# for chunk in chain.stream({"user": "大头", "char": "兔叽", "input": user_input}):
#      print(chunk)
#


