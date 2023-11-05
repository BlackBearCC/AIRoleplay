import llm_selector
import charact_selector
import prompt_manages


minimax = llm_selector.initialize_minimax()

tuji = charact_selector.initialize_tuji()

#初始化prompt
prompt = prompt_manages.initialize_prompt()
print(prompt)
