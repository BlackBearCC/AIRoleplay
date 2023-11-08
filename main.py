import llm_selector
import charact_selector
import prompt_manages
from langchain.prompts import ChatPromptTemplate


minimax = llm_selector.initialize_minimax()
qianfan = llm_selector.initialize_qianfan()

tuji = charact_selector.initialize_tuji()

prompt = prompt_manages.rolePlay()+tuji+prompt_manages.charactorStyle()+prompt_manages.plotDevelopment()
final_prompt = ChatPromptTemplate.from_template(prompt)

# chain = final_prompt | minimax
chain = final_prompt | qianfan


user_input = input()


for chunk in chain.stream({"user": "大头", "char": "兔叽", "input": user_input}):
    print(chunk)




