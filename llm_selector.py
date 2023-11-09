import logging
from langchain.llms import Minimax
from langchain.llms import QianfanLLMEndpoint

import os

os.environ["QIANFAN_AK"] = "FxhI5DprCvZQniOvLNwmp121"
os.environ["QIANFAN_SK"] = "E3TIfNHyMB8mF8rPwAYEUYMYBKqmxtdH"

def initialize_minimax():
    minimax= Minimax(minimax_api_key="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJOYW1lIjoidGVzdCIsIlN1YmplY3RJRCI6IjE2OTkwOTc5Njg0OTY1ODMiLCJQaG9uZSI6Ik1UZzJNVFkzTnpBeU1EUT0iLCJHcm91cElEIjoiMTY5OTA5Nzk2ODMwMzc3NiIsIlBhZ2VOYW1lIjoiIiwiTWFpbCI6Imxlb3p5MDgxOUBnbWFpbC5jb20iLCJDcmVhdGVUaW1lIjoiMjAyMy0xMS0wNCAyMToxODozMyIsImlzcyI6Im1pbmltYXgifQ.QEb4PUlFdewUeIIUbB1KvczqDNRv5mTb3XvWVj8J3kK6SDGN7qgtpjCmS7TBBWmJtKm3A3-0AG0BQHuiwcy_XzYNaS-Wp1heknIw1EWloCeZ82kndT1_zLM_592EepSjcq6Nb8oObClYtZnPhY9R0_VbEGpl533GvB35_KuCJb30eieLU9c2_mtSWkdri5IsZfzloZFOHiZiFhPtfdHHnFXZTZKXgnSkwfEmiimPuLHhaqZQUmkfWWEQ2FOSuDg79YTmtwK6OVAvlsNtIls0ymUmWIWk31M8XpXayL7aSfjli4TTbeYdOEidUTlCIpYwbOUS4Bu7bP-j9FwPcjcy3Q",minimax_group_id="1699097968303776")
    print("minimax initialized")
    return minimax

def initialize_qianfan():
    qianfan = QianfanLLMEndpoint(
        model="ERNIE-Bot-turbo"  # ¥0.008元/千tokens
        # model = "ERNIE-Bot" #0.012元/千tokens
    )
    return qianfan