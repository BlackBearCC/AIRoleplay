# charactor_zero_shot_agent.py


from typing import Sequence, Optional, List
from langchain.agents import ZeroShotAgent
from langchain.tools.base import BaseTool
from langchain.prompts import PromptTemplate


class CharactorZeroShotAgent(ZeroShotAgent):
    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        prefix: str = "提示前缀",
        suffix: str = "提示后缀",
        format_instructions: str = "你的自定义格式指令",
        input_variables: Optional[List[str]] = None,
    ) -> PromptTemplate:
        """
        创建一个自定义风格的提示模板。

        Args:
            tools: 代理将访问的工具列表，用于格式化提示。
            prefix: 放在工具列表前的字符串。
            suffix: 放在工具列表后的字符串。
            format_instructions: 自定义格式指令。
            input_variables: 最终提示将期望的输入变量列表。

        Returns:
            组装好的提示模板。
        """
        # 生成工具描述字符串
        tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        # 生成工具名称列表
        tool_names = ", ".join([tool.name for tool in tools])
        # 格式化指令
        format_instructions = format_instructions.format(tool_names=tool_names)
        # 组装模板
        template = "\n\n".join([prefix, tool_strings, format_instructions, suffix])
        # 设置输入变量
        if input_variables is None:
            input_variables = ["input", "agent_scratchpad"]
        return PromptTemplate(template=template, input_variables=input_variables)
