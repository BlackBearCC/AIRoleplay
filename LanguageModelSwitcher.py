import logging
import os
os.environ["QIANFAN_AK"] = "FxhI5DprCvZQniOvLNwmp121"
os.environ["QIANFAN_SK"] = "E3TIfNHyMB8mF8rPwAYEUYMYBKqmxtdH"
os.environ["OPENAI_API_KEY"] = "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"

class LanguageModelSwitcher:
    """
    这个类用于根据给定的model_type来切换不同的语言模型。
    初始化时，需要传入一个字符串类型的model_type，用于指定需要初始化的语言模型类型。
    """
    def __init__(self, model_type):
        self.model = None
        self.model_type = model_type
        self.switch_model(model_type)

    def switch_model(self,model_type):
        if model_type == "minimax":
            self.model = self.initialize_minimax()
        elif model_type == "qianfan":
            self.model = self.initialize_qianfan()
        elif model_type == "openai":
            self.model = self.initialize_Openai()
        else:
            raise ValueError("model_type not found")

    def initialize_minimax(self):
        from langchain.llms import Minimax
        minimax = Minimax(
            minimax_api_key="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJOYW1lIjoidGVzdCIsIlN1YmplY3RJRCI6IjE2OTkwOTc5Njg0OTY1ODMiLCJQaG9uZSI6Ik1UZzJNVFkzTnpBeU1EUT0iLCJHcm91cElEIjoiMTY5OTA5Nzk2ODMwMzc3NiIsIlBhZ2VOYW1lIjoiIiwiTWFpbCI6Imxlb3p5MDgxOUBnbWFpbC5jb20iLCJDcmVhdGVUaW1lIjoiMjAyMy0xMS0wNCAyMToxODozMyIsImlzcyI6Im1pbmltYXgifQ.QEb4PUlFdewUeIIUbB1KvczqDNRv5mTb3XvWVj8J3kK6SDGN7qgtpjCmS7TBBWmJtKm3A3-0AG0BQHuiwcy_XzYNaS-Wp1heknIw1EWloCeZ82kndT1_zLM_592EepSjcq6Nb8oObClYtZnPhY9R0_VbEGpl533GvB35_KuCJb30eieLU9c2_mtSWkdri5IsZfzloZFOHiZiFhPtfdHHnFXZTZKXgnSkwfEmiimPuLHhaqZQUmkfWWEQ2FOSuDg79YTmtwK6OVAvlsNtIls0ymUmWIWk31M8XpXayL7aSfjli4TTbeYdOEidUTlCIpYwbOUS4Bu7bP-j9FwPcjcy3Q",
            minimax_group_id="1699097968303776")
        logging.info("minimax initialized")
        return minimax

    def initialize_qianfan(self):
        from langchain.llms import QianfanLLMEndpoint
        logging.info("qianfan initialized")
        return QianfanLLMEndpoint(model="ERNIE-Bot-turbo")

    def initialize_Openai(self):
        from langchain.llms import OpenAI
        logging.info("openai initialized")
        return OpenAI()

