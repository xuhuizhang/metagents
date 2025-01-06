import os
import pandas as pd
from riskchat import (
    LLMList,
    GetPythonChain,
)
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

model = LLMList("glm4_p").chat_model(0.1, streaming=True)


def py_code_runner(code, df):
    response = GetPythonChain.python_exec(code, df)
    return response


class CreatePythonInput(BaseModel):
    python_code: str = Field(..., description="用户提供的符合Python3规范的代码")
    dataframe: list[str] = Field(..., description="用户提供数据集, 如`df1.csv`")
    python_run: str = Field(..., description="确认是否运行Python代码")


class PythonRunner(BaseTool):
    """
    这是一个负责执行Python代码的解释器。
    本工具需要用户提供符合Python3规范的代码和相应的分析数据集，缺一不可。
    如果用户没有提供这些信息，或者缺少一些信息，则提示用户提供对应的信息，直到信息完整。
    """
    name: str = "PythonRunner"
    description: str = """
    这是一个负责执行Python代码的解释器。
    本工具需要用户提供符合Python3规范的代码和相应的分析数据集，缺一不可。
    如果用户没有提供这些信息，或者缺少一些信息，则提示用户提供对应的信息，直到信息完整。
    """
    args_schema: Type[BaseModel] = CreatePythonInput

    def _run(self, python_code, dataframe, python_run):
        """
        开始运行Python代码
        """
        print(python_run)
        df = pd.read_csv(
            os.path.join(_PROJECT_PATH, 'docs', 'DataFrame_Files', dataframe)
        )
        py_response = py_code_runner(python_code, df)
        print(py_response)
        return py_response

    async def _arun(self, python_code, dataframe, python_run):
        """
        开始运行Python代码
        """
        print(python_run)
        df = pd.read_csv(dataframe)
        py_response = py_code_runner(python_code, df)
        print(py_response)
        return py_response


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
    llm_with_tools = model.bind_tools([PythonRunner()])
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}  # type: ignore


def python_supervisor_build():
    tool_node = ToolNode([PythonRunner()])
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


def python_supervisor_workflow():
    _graph = python_supervisor_build()
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
    python_supervisor_workflow()
