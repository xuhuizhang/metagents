from langchain_core.messages import AIMessage, HumanMessage
import os
import asyncio
import streamlit as st
import numpy as np
from load_db import save_conversation, save_message

user_icon = st.session_state.style['user_icon']
ai_icon = st.session_state.style['ai_icon']

def clear_chat_history():
    st.session_state.conversation_id = save_conversation('metagent_risk')
    st.session_state.messages = []
    st.session_state.training_history = []


@st.cache_resource
def graph_stream():
    from workflows import rag_supervisor_build
    return rag_supervisor_build()


# @st.cache_resource
def get_graph_stream():
    from streaming import GraphConversation
    return GraphConversation


llm = graph_stream()
graph_conver = get_graph_stream()(app=llm)

if query := st.chat_input(placeholder='请输入问题'):

    st.session_state.messages.append(HumanMessage(content=query))
    input_messages = (
        st.session_state.messages[-3:]
        if len(st.session_state.messages) >= 3 else st.session_state.messages
    )
    with st.chat_message('user', avatar=user_icon):
        st.markdown(query)
    with st.chat_message('ai', avatar=ai_icon):
        # print("="*20)
        # print("input_messages", input_messages)
        content = asyncio.run(
            graph_conver.stream_conversation({"messages": input_messages})
        )
    if content is not None:
        st.session_state.messages.append(AIMessage(content=content))
        save_message(st.session_state.conversation_id, query, content, 'Risk Policy')




st.sidebar.button(
    'New Chat',
    type="primary",
    use_container_width=True,
    on_click=clear_chat_history
)
