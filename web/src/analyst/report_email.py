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



# if 'messages' not in st.session_state:
#     st.session_state.messages = []

# for message in st.session_state.messages:
#     if isinstance(message, HumanMessage):
#         with st.chat_message('user', avatar=user_icon):
#             st.write(message.content)
#     elif isinstance(message, AIMessage):
#         with st.chat_message('ai', avatar=ai_icon):
#             if isinstance(message.content, Figure):
#                 st.write(message.content)
#             elif 'SELECT' in message.content and 'FROM' in message.content:
#                 st.code(message.content, language='sql')
#             elif 'import' in message.content:
#                 st.code(message.content, language='python')
#             else:
#                 st.write(message.content)

@ st.cache_resource
def graph_stream():
    from workflows import send_email_build
    return send_email_build()


@ st.cache_resource
def get_graph_stream():
    from streaming import GraphConversation
    return GraphConversation


llm = graph_stream()
graph_conver = get_graph_stream()(app=llm)

if query := st.chat_input(placeholder='请输入问题'):

    st.session_state.messages.append(HumanMessage(content=query))
    input_messages = (
        st.session_state.messages[-7:]
        if len(st.session_state.messages) >= 7 else st.session_state.messages
    )
    with st.chat_message('user', avatar=user_icon):
        st.markdown(query)
    with st.chat_message('ai', avatar=ai_icon):
        content = asyncio.run(
            graph_conver.stream_conversation({"messages": input_messages}))
    if content is not None:
        st.session_state.messages.append(AIMessage(content=content))
        save_message(st.session_state.conversation_id, query, content, 'Reporting & Email')




st.sidebar.button(
    'New Chat',
    type="primary",
    use_container_width=True,
    on_click=clear_chat_history
)
