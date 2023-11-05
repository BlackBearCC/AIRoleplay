import llm_selector
import charact_selector
import prompt_manages
from langchain.prompts import ChatPromptTemplate


minimax = llm_selector.initialize_minimax()

tuji = charact_selector.initialize_tuji()

prompt = prompt_manages.rolePlay()+tuji+prompt_manages.charactorStyle()+prompt_manages.plotDevelopment()
final_prompt = ChatPromptTemplate.from_template(prompt)

chain = final_prompt | minimax


user_input = input()

while user_input != "exit":
    print(1)
    for chunk in chain.stream({"user": "大头", "char": "兔叽", "input": user_input}):
        print(chunk)
    user_input = input()



