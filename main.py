import llm_selector
import charact_selector
import prompt_manages
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import get_openai_callback

minimax = llm_selector.initialize_minimax()
qianfan = llm_selector.initialize_qianfan()

tuji = charact_selector.initialize_tuji()

prompt = prompt_manages.rolePlay()+tuji+prompt_manages.charactorStyle()+prompt_manages.plotDevelopment()
final_prompt = ChatPromptTemplate.from_template(prompt)

# chain = final_prompt | minimax
chain = final_prompt | qianfan


user_input = input()


# for chunk in chain.stream({"user": "大头", "char": "兔叽", "input": user_input}):
#     print(chunk)



with get_openai_callback() as callback:
    for chunk in chain.stream({"user": "大头", "char": "兔叽", "input": user_input}):
        print(chunk)

    print(callback)
# with get_openai_callback() as callback:
#     for chunk in chain.stream({"user": "大头", "char": "兔叽", "input": user_input}, callback=callback):
#         print(chunk)

