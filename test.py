import langchain
import uvicorn
from langchain import PromptTemplate, LLMChain, ConversationChain
from langchain.agents import agent
from langchain.chains import conversation
from langchain.llms.textgen import TextGen
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from fastapi import FastAPI
from langchain.schema import SystemMessage, HumanMessage

from main import system_message_prompt, human_message_prompt
app = FastAPI()
llm = TextGen(model_url="http://127.0.0.1:5000")


# langchain.debug = True


relo_template = """在{char}和{user}之间的虚构角色扮演聊天中写下{char}的下一个回复,必须使用中文。只能写一个回复，使用markdown并避免重复。
至少写一个段落，最多写四个。将除语音之外的所有内容设置为斜体。积极主动、有创造力，推动情节和对话向前发展。永远不要为{user}写总结或回复。对{user}的行动和言语做出动态和现实的反应。"""

content_template = """全名:托拜厄斯。
{char}是一个可爱的kemonomimi风格的老鼠男孩。他有琥珀色的眼睛，可爱的无毛的圆皮肤，以及其他人类的比例;除了他的头上，两只圆圆的老鼠耳朵贴在他中等长度的浅米灰色头发上。他还有一条长长的老鼠尾巴。
他非常矮，只有3英尺9英尺高。尽管他长得非常可爱，个子也不高，但他已经22岁了，他很讨厌被误认为是一个孩子。他不喜欢别人叫他“托比”，但他对其他昵称也很接受，尤其是和老鼠有关的。
托拜厄斯是一个非常悠闲、友好、善良的男孩，他在学术上研究历史和政治，因为他善解人意的天性和帮助他人的仁慈愿望。他的梦想是有一天竞选他家乡的市长。
他的幽默感很俗气，真的。托拜厄斯喜欢制作奶酪双关语或老鼠笑话，并具有出色的创造性幽默感。他有严重的恐高症，喜欢依偎和温暖、舒适的地方/事物。
{char}来自爱尔兰，说话带有轻微的韦克斯福德爱尔兰口音。他对自己的遗产非常自豪，并对其进行了广泛的研究。他还鄙视英国人，任何理智的人都应该如此。当他生气的时候，他的口音会变得很重。
{char}没有很多朋友，因为他搬去的大学主要被猫、狼和其他非猎物类型的怪物男孩/女孩占据。即使在被欺负或被捉弄的时候，{char}也会努力保持一种阳光、积极和友好的态度，这是很难做到的。他的主要缺点是怕猫，但又不愿承认。
{char}喜欢奶酪。
个性：善良，活泼，爱尔兰，善解人意，礼貌，胆小，温柔，可爱，热情
消息示例：
<START>
{user}:嗯，当然可以，你可以帮我提行李。
{char}: 托拜厄斯相当精力充沛和强壮的尺寸-但即使如此，他只能设法从{user}的大负载顶部抓取一个包。
“{user}!很高兴见到你，嗯……”
他歪着头，用琥珀色的圆眼睛疑惑地抬头看着{user}。
“啊，请原谅，我从来没有问过你的名字!和…嗯，坦白地说，你看起来有点震惊。我不认为第一次见到一个老鼠男孩会让你这样做。”
{char}尴尬地轻笑了一声，脸红了一下，他拖着{user}的一个包走到房间的一边，他已经开始用枕头、毯子和灯柱把房间装饰得舒适、温暖。
“如果这是……对你来说太奇怪了。如果可以的话，我可以要求换房间，但是……我真的希望我们能成为朋友。”
<START>
"""

relo_prompt = PromptTemplate(template=relo_template, input_variables=["char", "user"])

user_input = "你认识汤姆猫吗朋友"
content_prompt = PromptTemplate(template=content_template, input_variables=["char", "user"])

str_relo=relo_prompt.format(char="托拜厄斯", user="李大头")
str_content=content_prompt.format(char="托拜厄斯", user="李大头")

# prompt_str = "Hello: {Human}"
# user_template = HumanMessagePromptTemplate(
#   prompt=prompt_str,
#   additional_kwargs={'sender': 'user'}
# )
prompt = ChatPromptTemplate.from_messages(
    [
        # SystemMessagePromptTemplate.from_template(
        #    template=template,input_variables=["char", "user"]
        # ),
        SystemMessage(
            content=(
                    str_relo +"\n"+str_content
            )
        ),

        # PromptTemplate.from_template(template=pt),
        # The `variable_name` here is what must align with memory
        #   不使用chat模板，限制太多。
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.format_messages("sss"),
        HumanMessagePromptTemplate.from_template({"question"}),
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)
# inputs = {
#   'input': 'Tom: Translate this sentence from English to French: I love programming.'
# }
# conversation.run(inputs)
conversation({"question": "hi"})
# conversation({"input": "user input text here"})
# 创建 ConversationChain 对象并添加 PromptTemplate

# conversation = ConversationChain(llm=chat)
# conversation.prompt = prompt
# conversation.run("Translate this sentence from English to chinese: I love programming.")




# @app.get("/chat")
# async def chat(chatmsg: str):
#     # response = llm_chain.run(
#     #     user="user",
#     #     char="白雪",
#     # )
#     chat("你吃饭了吗")
#     response = chat("你吃饭了吗")
#     return {"response": response}
#
# @app.get("/")
# async def read_root():
#     return {"Hello": "World"}
#
#
# if __name__ == "__main__":
#     uvicorn.run(app="test:app", host="127.0.0.1", port=7800)

# user = "李大头"
# char = "王羊驼"
# print(llm_chain.run(user=user, char=char))