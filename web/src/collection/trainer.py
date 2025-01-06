from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os
import streamlit as st
from matplotlib.figure import Figure
from pathlib import Path
from typing import Literal, Tuple
import pandas as pd
import random
from web.src.collection.scenarios import Scenario, EVALUATION_PROMPT

_SRC_PATH = Path(__file__).resolve().parents[1]


user_icon = st.session_state.style['user_icon']
ai_icon = st.session_state.style['ai_icon']

if 'training_history' not in st.session_state:
    st.session_state.training_history = []
    

if 'random_number' not in st.session_state:
    st.session_state.random_number = 10

for message in st.session_state.training_history:
    if isinstance(message, HumanMessage):
        with st.chat_message('user', avatar=user_icon):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message('ai', avatar=ai_icon):
            st.write(message.content)


@st.cache_data
def account_info() -> pd.DataFrame:
    _df = pd.read_excel(
        os.path.join(_SRC_PATH, 'collection', 'client_info.xlsx')
    )
    return _df


@st.cache_resource
def select_scenario(
    level: Literal['EASY', 'MEDIUM', 'HARD'],
    random_number: int,
) -> Tuple[str, str]:
    level_mapping = {
        'EASY': Scenario.EASY,
        'MEDIUM': Scenario.MEDIUM,
        'HARD': Scenario.HARD

    }

    try:
        scenario = level_mapping[level]
    except KeyError:
        raise ValueError(f"无效的级别: {level}. 请选择 'EASY', 'MEDIUM', 或 'HARD'.")

    df = account_info()

    selected_client = df[
        df['level'] == scenario.name.lower()
    ].sample(n=1, random_state=random_number).iloc[0]

    intro = scenario.intro.format(
        client_name=selected_client['client_name'],
        amt_outstanding_total=selected_client['amt_outstanding_total'],
        cnt_days_past_due=selected_client['cnt_days_past_due_tolerance']
    )

    content = random.choice(scenario.content_list).format(
        client_name=selected_client['client_name'],
        amt_outstanding_total=selected_client['amt_outstanding_total'],
        cnt_days_past_due=selected_client['cnt_days_past_due_tolerance']
    )

    return intro, content


def clear_chat_history():
    st.session_state.messages = []
    st.session_state.training_history = []
    st.session_state.random_number = random.randint(0, 100)


level = st.sidebar.selectbox(
    label="Choose Training Level",
    options=[Scenario.EASY.name, Scenario.MEDIUM.name, Scenario.HARD.name],
    index=0,
    key="scenarioRadio",
    on_change=clear_chat_history,
)

level_intro, level_content = select_scenario(
    level, st.session_state.random_number  # type: ignore
)

st.sidebar.write(level_intro)

sys_message = f'''  
任务：请你仔细认真阅读instruction中的信息。按照instruction中的信息中的指示扮演好逾期客户的角色，出色的完成对话任务。
<instruction> {level_content} </instruction>
'''


@st.cache_resource
def llm_trainer():
    from riskchat import LLMList
    llm_chat = LLMList('glm4_p').chat_model(0.9, streaming=True)
    return llm_chat


llm = llm_trainer()


st.sidebar.button(
    'New Chat',
    type="primary",
    use_container_width=True,
    on_click=clear_chat_history
)


if query := st.chat_input(placeholder='请输入问题'):
    st.session_state.training_history.append(HumanMessage(content=query))
    with st.chat_message('user', avatar=user_icon):
        st.markdown(query)
    with st.chat_message('ai', avatar=ai_icon):
        messages = (
            [SystemMessage(content=sys_message)] +
            st.session_state.training_history
        )
        answer = st.write_stream(llm.stream(messages))

    if answer is not None:
        st.session_state.training_history.append(AIMessage(content=answer))

if st.sidebar.button(
    'Training Evaluation',
    type="secondary",
    use_container_width=True,
):
    if len(st.session_state.training_history):
        messages = (
            st.session_state.training_history[0::2] +
            [HumanMessage(content=EVALUATION_PROMPT)]
        )
        st.write_stream(llm.stream(messages))
    else:
        st.warning('找不到对话')
