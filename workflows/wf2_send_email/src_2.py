# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# import smtplib
from riskchat import LLMList
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from typing import Literal, Type, Union
import os
from datetime import datetime
from pathlib import Path
from workflows.tools import send_email
_SRC_PATH = Path(__file__).resolve().parents[0]
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


model = LLMList("glm4_p").chat_model(0.1, streaming=True)
# model = LLMList("deepseek_chat").chat_model(0.2, streaming=True)


# def send_email(subject: str, receivers: Union[list[str], str], content: str, sender: str = "LLM"):
#     """给指定的邮箱发送邮件"""
#     message = MIMEMultipart("alternative")
#     message["Subject"] = subject
#     message["To"] = "; ".join(receivers) if isinstance(
#         receivers, list) else receivers
#     message["From"] = sender
#     message.attach(MIMEText(content, "html", _charset="utf-8"))
#     try:
#         with smtplib.SMTP(host="relay.homecredit.cn", port=25) as server:
#             server.sendmail(sender, receivers, message.as_string())
#             print("邮件发送成功.")
#     except Exception as e:
#         print(f"邮件发送失败: {e}")
#     else:
#         print("邮件发送成功.")
#     finally:
#         pass


class CreateEmailInput(BaseModel):
    subject: str = Field(description="邮件主题")
    content: str = Field(description="邮件的内容")
    receivers: Union[list[str], str] = Field(description="邮件的收件人")
    confirm: str = Field(description="是否确认发送邮件")


class SendEmail(BaseTool):
    """
    这是一个发送邮件的工具，需要用户提供邮件的主题，收件人，邮件内容以及最后的确认信息，缺一不可。
    如果用户没有提供这些信息，或者缺少一些信息，则提示用户提供对应的信息，直到信息完整。
    注意：发送邮件前，请先将邮件内容转换为HTML格式，然后才能发送。
    最后，邮件工具调用后，请结束工具调用并返回调用成功通知。
    """

    name: str = "SendEmail"
    description: str = """
    这是一个发送邮件的工具，需要用户提供邮件的主题，收件人，邮件内容以及最后的确认信息，缺一不可。
    如果用户没有提供这些信息，或者缺少一些信息，则提示用户提供对应的信息，直到信息完整。
    注意：发送邮件前，请先将邮件内容转换为HTML格式，然后才能发送。
    最后，邮件工具调用后，请结束工具调用并返回调用成功通知。
    """
    args_schema: Type[BaseModel] = CreateEmailInput

    def _run(
        self, subject: str, receivers: Union[list[str], str], content: str, confirm: str
    ) -> str:
        """
        邮件发送
        """
        if confirm:
            send_email(subject, receivers, content)
        return "邮件发送成功！"

    async def _arun(
        self, subject: str, receivers: Union[list[str], str], content: str, confirm: str
    ) -> str:
        """
        异步邮件发送
        """
        if confirm:
            send_email(subject, receivers, content)
        return "邮件发送成功！"


def should_continue(state: MessagesState) -> Literal["tools", "end"]:
    """
    判断是否需要调用发送邮件的工具。
    如果调用工具则返回tools, 如果不需要发送邮件则返回END
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
    llm_with_tools = model.bind_tools([SendEmail()], tool_choice='any')
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}  # type: ignore


def send_email_build():
    tool_node = ToolNode([SendEmail()])
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


def send_email_workflow():
    _graph = send_email_build()
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
                    {"messages": messages}, config={"configurable": {"thread_id": "1"}}
                )
                messages.append(
                    AIMessage(content=response["messages"][-1].content))
                file.write(f"\n\n```\n{response}\n```")
                print("AI:", response["messages"][-1].content)
                print("log:", response)


if __name__ == "__main__":
    send_email_workflow()
