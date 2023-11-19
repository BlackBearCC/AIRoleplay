from typing import Union

from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
import re


class CustomOutputParser(AgentOutputParser):
    """Parses tool invocations and final answers in XML format.

    Expects output to be in one of two formats.

    If the output signals that an action should be taken,
    should be in the below format. This will result in an AgentAction
    being returned.

    ```
    <tool>search</tool>
    <tool_input>what is 2 + 2</tool_input>
    ```

    If the output signals that a final answer should be given,
    should be in the below format. This will result in an AgentFinish
    being returned.

    ```
    <final_answer>Foo</final_answer>
    ```
    """

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "</tool>" in text:
            tool, tool_input = text.split("</tool>")
            _tool = tool.split("<tool>")[1]
            try:
                _tool_input = tool_input.split("<tool_input>")[1]
            except:
                _tool_input = "我没法评估"
            # _tool_input = tool_input.split("<tool_input>")[1]
            if "</tool_input>" in _tool_input:
                _tool_input = _tool_input.split("</tool_input>")[0]
            return AgentAction(tool=_tool, tool_input=_tool_input, log=text)
        elif "<final_answer>" in text:
            _, answer = text.split("<final_answer>")
            if "</final_answer>" in answer:
                answer = answer.split("</final_answer>")[0]
            return AgentFinish(return_values={"output": answer}, log=text)
        else:

            return AgentFinish(return_values={"output": text}, log=text)

    def get_format_instructions(self) -> str:
        raise NotImplementedError

    @property
    def _type(self) -> str:
        return "charactor-agent"

    #
    # def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
    #     # Check if agent should finish
    #     if "Final Answer:" in llm_output:
    #         return AgentFinish(
    #             # Return values is generally always a dictionary with a single `output` key
    #             # It is not recommended to try anything else at the moment :)
    #             return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
    #             log=llm_output,
    #         )
    #     # Parse out the action and action input
    #     regex = r"Action:\s*(.*?)\nAction Input:\s*(.*)"
    #     match = re.search(regex, llm_output, re.DOTALL)
    #     if match:
    #         action = match.group(0).strip()
    #         action_input = match.group(1).strip()
    #         print(f"Action: {action}")
    #         print(f"Action Input: {action_input}")
    #     else:
    #         action = "Chat"
    #         action_input = "无法解析输出"
    #     return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)