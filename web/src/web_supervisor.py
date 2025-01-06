from langchain_core.messages import AIMessage, HumanMessage
import os
import asyncio
import streamlit as st
import numpy as np
from PIL import Image
from matplotlib.figure import Figure
from pathlib import Path
_SRC_PATH = Path(__file__).resolve().parents[1]
st.set_page_config(page_title='Welcome to RiskChat', page_icon='💬')
user_icon = Image.open(os.path.join(_SRC_PATH, 'public', 'h.png'))
ai_icon = Image.open(os.path.join(_SRC_PATH, 'public', 'ai.png'))


with open(os.path.join(_SRC_PATH, 'styles', 'global.css')) as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message('user', avatar=np.array(user_icon)):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message('ai', avatar=np.array(ai_icon)):
            if isinstance(message.content, Figure):
                st.write(message.content)
            else:
                st.write(message.content)


def clear_chat_history():
    st.session_state.messages = []


@st.cache_resource
def graph_stream():
    from workflows import supervisor_build
    return supervisor_build()


@st.cache_resource
def get_graph_stream():
    from streaming import GraphConversation
    return GraphConversation


llm = graph_stream()
graph_conver = get_graph_stream()(app=llm)

if query := st.chat_input(placeholder='请输入问题'):

    st.session_state.messages.append(HumanMessage(content=query))
    input_messages = (
        st.session_state.messages[-9:]
        if len(st.session_state.messages) >= 9 else st.session_state.messages
    )
    with st.chat_message('user', avatar=np.array(user_icon)):
        st.markdown(query)
    with st.chat_message('ai', avatar=np.array(ai_icon)):
        # print("="*20)
        # print("input_messages", input_messages)
        content = asyncio.run(
            graph_conver.stream_conversation({"messages": input_messages})
        )
    if content is not None:
        st.session_state.messages.append(AIMessage(content=content))

st.sidebar.button('清除历史消息', on_click=clear_chat_history)

# with st.sidebar:
#     for message in st.session_state.messages:
#         if isinstance(message, HumanMessage):
#             st.button(str(message.content))
