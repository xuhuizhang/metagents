import os
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import json
from riskchat import (
    LLMList,
    RagLLM,
    GetSQLChain,
    ConnectSQL,
    _BACKGROUND_DOC,
    _DB_SCHEMA_DOC,
    rag_rerank_doc
)
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph, MessagesState
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal, Type
from datetime import datetime
from workflows.tools import send_email

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

model = LLMList("glm4_p").chat_model(0.1, streaming=True)
# model = LLMList("glm4_alltools").chat_model(0.1, streaming=True)
# model = LLMList("deepseek_chat").chat_model(0.1, streaming=True)


def es_rag(id: int, question: str):
    data = {
        "text": question,
        "mode": "3",
        "top_k": 20,
        "rerank_top_k": 10,
        "threshold": 0
    }
    retriever_docs = requests.post(
        f'http://10.26.9.148:31817/vector_stores/{id}/search',
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




class CreateUWInput(BaseModel):
    question: str = Field(..., description="用户提出的有关UW信贷审批业务的问题")


class UWSearch(BaseTool):
    """
    这是一个有关UW信贷审批业务增强检索（RAG）的工具。
    本工具结合用户提供的UW信贷审批业务相关的问题，从而进行RAG的使用。
    """
    name: str = "UWSearch"
    description: str = """
    这是一个有关UW信贷审批业务增强检索（RAG）的工具。
    本工具结合用户提供的UW信贷审批业务相关的问题，从而进行RAG的使用。
    """
    args_schema: Type[BaseModel] = CreateUWInput

    def _run(self, question):
        """
        开始进行RAG检索
        """
        answer = es_rag(62, question)
        print(answer)
        return answer

    async def _arun(self, question):
        """
        开始进行RAG检索
        """
        answer = es_rag(62, question)
        print(answer)
        return answer


class CreateDBSchemaInput(BaseModel):
    question: str = Field(..., description="用户提出的有关数据库表结构Schema的问题")


class DBSchemaSearch(BaseTool):
    """
    这是一个专用于检索数据库表结构Schema的RAG工具。
    本工具需要用户检索数据库表格的问题，才能进行RAG的使用。
    """
    name: str = "DBSchemaSearch"
    description: str = """
    这是一个专用于检索数据库表结构Schema的RAG工具。
    本工具需要用户检索数据库表格的问题，才能进行RAG的使用。
    """
    args_schema: Type[BaseModel] = CreateDBSchemaInput

    def _run(self, question):
        """
        开始进行RAG检索
        """
        answer = es_rag(63, question)
        print(answer)
        return answer

    async def _arun(self, question):
        """
        开始进行RAG检索
        """
        answer = es_rag(63, question)
        print(answer)
        return answer

    
# class CreateEmailInput(BaseModel):
#     subject: str = Field(description="邮件主题")
#     content: str = Field(description="邮件的内容")
#     receivers: list[str] = Field(description="邮件的收件人")
#     confirm: str = Field(description="是否确认发送邮件")


# class SendEmail(BaseTool):
#     """
#     这是一个发送邮件的工具，需要用户提供邮件的主题，收件人，邮件内容以及最后的确认信息，缺一不可。
#     如果用户没有提供这些信息，或者缺少一些信息，则提示用户提供对应的信息，直到信息完整。
#     """

#     name: str = "SendEmail"
#     description: str = """
#     这是一个发送邮件的工具，需要用户提供邮件的主题，收件人，邮件内容以及最后的确认信息，缺一不可。
#     如果用户没有提供这些信息，或者缺少一些信息，则提示用户提供对应的信息，直到信息完整。
#     """
#     args_schema: Type[BaseModel] = CreateEmailInput

#     def _run(
#         self, subject: str, receivers: list[str], content: str, confirm: str
#     ) -> str:
#         """
#         邮件发送
#         """
#         if confirm:
#             send_email(subject, receivers, content)
#         return "邮件发送成功！"

#     async def _arun(
#         self, subject: str, receivers: list[str], content: str, confirm: str
#     ) -> str:
#         """
#         异步邮件发送
#         """
#         if confirm:
#             send_email(subject, receivers, content)
#         return "邮件发送成功！"
    
    
    

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
    llm_with_tools = model.bind_tools(
        [UWSearch(), DBSchemaSearch()],
        tool_choice="auto"
    )
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}  # type: ignore


def rag_supervisor_build():
    tool_node = ToolNode(
        [UWSearch(), DBSchemaSearch()]
    )
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


# def rag_supervisor_workflow():
#     _graph = rag_supervisor_build()
#     messages = []
#     stop = True
#     while stop:
#         print("*" * 20)
#         # print(messages)
#         # print('*'*20)
#         user_input = input("Human：")
#         if user_input == "exit":
#             stop = False
#         else:
#             messages.append(HumanMessage(content=user_input))
#             response = _graph.invoke(
#                 {"messages": messages[-9:] if len(messages) >= 9 else messages}, config={"configurable": {"thread_id": "1"}}
#             )
#             messages.append(
#                 AIMessage(content=response["messages"][-1].content))
#             file.write(f"\n\n```\n{response}\n```")
#             print("AI:", response["messages"][-1].content)
#             print("log:", messages)


# if __name__ == "__main__":
#     rag_supervisor_workflow()
