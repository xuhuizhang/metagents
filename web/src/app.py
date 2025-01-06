import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import os, hmac, io, sys
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage
from load_db import save_conversation

_SRC_PATH = Path(__file__).resolve().parents[1]
_PROJECT_PATH = Path(__file__).resolve().parents[2]

st.set_page_config(page_title='Welcome to METAGENT', page_icon='💬')
st.logo(os.path.join(_SRC_PATH, 'public', 'logo.svg'))

if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = save_conversation('metagent_risk')

def check_password():

    def password_entered():
        """
        Checks whether a password entered by the user is correct.
        """
        if hmac.compare_digest(st.session_state['password'], st.secrets['password']):
            st.session_state['password_correct'] = True
            del st.session_state['password']
        else:
            st.session_state['password_correct'] = False

    # Return True if the password is validated.
    if st.session_state.get('password_correct', False):
        return True

    form = st.form(key='auth', border=False)
    form.title('欢迎使用 :violet[METAGENT]', anchor=False)
    form.divider()
    form.text_input('用户名: ', key='username')
    form.text_input('密码: ', type='password', key='password')
    form.form_submit_button('登录', on_click=password_entered, type='primary')
    if 'password_correct' in st.session_state:
        st.error('😕 密码不正确!')
    return False

@st.cache_resource
def load_human_icons():
    with open(os.path.join(_SRC_PATH, 'public', 'h.png'), 'rb') as f:
        user_icon_bytes = f.read()
    user_icon = Image.open(io.BytesIO(user_icon_bytes))
    return user_icon


@st.cache_resource
def load_ai_icons():
    with open(os.path.join(_SRC_PATH, 'public', 'ai.png'), 'rb') as f:
        ai_icon_bytes = f.read()
    ai_icon = Image.open(io.BytesIO(ai_icon_bytes))
    return ai_icon



@st.cache_resource
def page_style():
    with open(os.path.join(_SRC_PATH, 'styles', 'global.css')) as f:
        css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        img[data-testid = "stLogo"] {
                height: 4rem;
            }
        </style >
        """,
        unsafe_allow_html=True
    )


page_style()

user_icon = load_human_icons()
ai_icon = load_ai_icons()

if 'style' not in st.session_state:
    st.session_state.style = {
        'user_icon': np.array(user_icon),
        'ai_icon': np.array(ai_icon),
    }


if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message('user', avatar=user_icon):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message('ai', avatar=ai_icon):
            st.write(message.content)

def add_pages():
    pg = st.navigation(
        [
            
                st.Page(
                    "./documents/law_regulation.py", title="Law Regulation"
                ),
                st.Page(
                    "./documents/hcc_policy.py", title="HCC Policy(Internal)"
                ),
                st.Page("./analyst/uw_policy.py", title="Risk Policy"),
                st.Page(
                    "./analyst/report_email.py", title="Reporting & Email"
                )

            ]

    )
    pg.run()

def main():

    if not check_password():
        st.stop()
    else:
        add_pages()


main()
