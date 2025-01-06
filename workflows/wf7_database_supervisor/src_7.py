import os
import pandas as pd
from riskchat import LLMList, RagLLM, GetSQLChain, ConnectSQL, _BACKGROUND_DOC
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph, MessagesState
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal, Type

from pathlib import Path

_SRC_PATH = Path(__file__).resolve().parents[0]
_PROJECT_PATH = Path(__file__).resolve().parents[2]


model = LLMList("glm4_p").chat_model(0.1, streaming=True)
# model = LLMList("deepseek_chat").chat_model(0.1, streaming=True)


def sql_code_runner(code):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    _df = GetSQLChain.sql_to_pandas(code)
    _path = os.path.join('docs', 'DataFrame_Files', f'database_{now}.csv')
    _df.to_csv(_path)
    response = f"执行完毕，请您前往路径{_path}进行查看"
    return response


class CreateDBInput(BaseModel):
    sql_code: str = Field(..., description="提供的符合MySQL规范的数据库代码")
    sql_run: str = Field(..., description="确认是否启动数据库查询")


class DBRunner(BaseTool):
    """
    这是一个执行数据库查询的工具，可以运行数据库代码，进行数据库的查询操作。
    本工具需要用户提供符合MySQL规范的代码，和用户明确启动数据库查询的指令，缺一不可。
    如果用户没有提供这些信息，或者缺少一些信息，则提示用户提供对应的信息，直到信息完整。
    """
    name: str = "DBRunner"
    description: str = """
    这是一个执行数据库查询的工具，可以运行数据库代码，进行数据库的查询操作。
    本工具需要用户提供符合MySQL规范的代码，和用户明确启动数据库查询的指令，缺一不可。
    如果用户没有提供这些信息，或者缺少一些信息，则提示用户提供对应的信息，直到信息完整。
    """
    args_schema: Type[BaseModel] = CreateDBInput

    def _run(self, sql_code, sql_run):
        """
        执行数据库查询代码
        """
        print(sql_run)
        df_response = sql_code_runner(sql_code)
        print(df_response)
        return df_response

    async def _arun(self, sql_code, sql_run):
        """
        执行数据库查询代码
        """
        print(sql_run)
        df_response = sql_code_runner(sql_code)
        print(df_response)
        return df_response


def should_continue(state: MessagesState) -> Literal["tools", "end"]:
    """
    判断是否需要调用查询数据库的工具。
    如果需要进行数据库查询，则返回tools, 否则返回END
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
    llm_with_tools = model.bind_tools([DBRunner()])
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}  # type: ignore


def database_supervisor_build():
    tool_node = ToolNode([DBRunner()])
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


def database_supervisor_workflow():
    _graph = database_supervisor_build()
    messages = []
    with open(os.path.join(_SRC_PATH, "workflow_log.md"), "w", encoding="utf-8",) as file:
        file.write(
            f"# workflow_graph\n\n```mermaid\n{_graph.get_graph().draw_mermaid()}\n```"
        )
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
                file.write(f"\n\n```\n{response}\n```")
                print("AI:", response["messages"][-1].content)
                print("log:", messages)


if __name__ == "__main__":
    database_supervisor_workflow()
