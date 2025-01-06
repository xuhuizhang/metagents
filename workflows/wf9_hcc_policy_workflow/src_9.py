import os
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import json
from riskchat import LLMList
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph, MessagesState
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal, Type
from datetime import datetime
from pathlib import Path

_SRC_PATH = Path(__file__).resolve().parents[0]
_PROJECT_PATH = Path(__file__).resolve().parents[2]
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")



model = LLMList("glm4_p").chat_model(0.2, streaming=True)
# model = LLMList("deepseek_chat").chat_model(0.3, streaming=True)


# def send_rag(question):
#     content = []
#     _graph = rag_tool_build()
#     for _answer in _graph.stream(
#         {"messages": [HumanMessage(content=question)]},
#         {"recursion_limit": 3}
#     ):
#         for key, value in _answer.items():
#             if 'messages' in value:
#                 content.append(value['messages'][-1])
#     print(content)
#     return content

def es_rag(question):
    data = {
        "text": question,
        "mode": "3",
        "top_k": 20,
        "rerank_top_k": 10,
        "threshold": 0
    }
    retriever_docs = requests.post(
        'http://10.26.9.148:31817/vector_stores/58/search',
        auth=HTTPBasicAuth('hcc-rag', '45355$%Dff'),
        data=json.dumps(data)
    ).json()
    docs = ''
    if retriever_docs.get('data'):
        for x in retriever_docs['data']['retriever_doc']:
            docs = docs + '\n' + x['content'] + '\n'
        return docs
    else:
        return '没有收到相关信息'


class CreateInternalInput(BaseModel):
    question: str = Field(..., description="用户提出的涉及消费金融公司内部业务或政策方面的问题")


class InternalSearch(BaseTool):
    """
    这是一个涉及消费金融公司内部业务或政策的增强检索（RAG）的工具。
    本工具结合用户涉及消费金融公司内部业务或政策问题，从而进行RAG的使用。
    """
    name: str = "InternalSearch"
    description: str = """
    这是一个涉及消费金融公司内部业务或政策的增强检索（RAG）的工具。
    本工具结合用户涉及消费金融公司内部业务或政策问题，从而进行RAG的使用。
    """
    args_schema: Type[BaseModel] = CreateInternalInput

    def _run(self, question):
        """
        开始进行RAG检索
        """
        answer = es_rag(question)
        print(answer)
        return answer

    async def _arun(self, question):
        """
        开始进行RAG检索
        """
        answer = es_rag(question)
        print(answer)
        return answer


def should_continue(state: MessagesState) -> Literal["tools", "end"]:
    """
    判断是否需要调用工具，如果调用工具，则返回tools, 否则返回END
    """
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:  # type: ignore
        return "tools"
    return "end"


def call_llm(state: MessagesState) -> MessagesState:
    """
    调用大模型
    """
    messages = state["messages"]
    llm_with_tools = model.bind_tools([InternalSearch()], tool_choice="auto")
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}  # type: ignore


def hcc_policy_build():
    tool_node = ToolNode([InternalSearch()])
    workflow = StateGraph(MessagesState)
    workflow.add_node("llm", call_llm)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("llm")
    workflow.add_conditional_edges(
        "llm", should_continue, {"end": END, "tools": "tools"}
    )
    workflow.add_edge("tools", "llm")

    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    graph = workflow.compile()
    return graph


def hcc_policy_workflow():
    _graph = hcc_policy_build()
    messages = []
    stop = True
    while stop:
        print("*" * 20)
        # print(messages)
        # print('*'*20)
        user_input = input("Human：")
        if user_input == "exit":
            stop = False
        else:
            messages.append(HumanMessage(content=user_input))
            response = _graph.invoke(
                {"messages": messages[-9:] if len(messages) >= 9 else messages}, config={"configurable": {"thread_id": "1"}}
            )
            messages.append(
                AIMessage(content=response["messages"][-1].content))
            print("AI:", response["messages"][-1].content)
            print("log:", messages)


if __name__ == "__main__":
    hcc_policy_workflow()
