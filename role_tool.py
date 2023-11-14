from langchain.tools import BaseTool

#知识类工具
class KnowledgeTool(BaseTool):
    name = "Konwledge"
    description = "It is very useful when you need to reply from existing knowledg"

    def _run(self, query: str) -> str:
        return "我喜欢吃红烧肉"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")


class ActTool(BaseTool):
    name = "兴趣"
    description = "It is very useful when you need to take 兴趣."

    def _run(self, query: str) -> str:
        return "我喜欢打扑克牌"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")