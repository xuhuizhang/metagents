import streamlit as st
import time


class StreamlitAssistantAnswer:
    def __init__(self) -> None:
        self.tokens_area = st.empty()
        self.tokens_stream = ''

    def re_render_answer(self, token: str) -> None:
        self.tokens_stream += token
        self.tokens_area.markdown(self.tokens_stream)

    def confirm_answer(self, message: str) -> None:
        self.tokens_area.markdown(message)


class GraphConversation:
    def __init__(self, app):
        self.app = app

    async def stream_conversation(self, messages):
        assistant_answer = StreamlitAssistantAnswer()
        async for event in self.app.astream_events(messages, version='v1'):
#             print('all_event: ', event)
            kind = event['event']

            if kind == 'on_tool_start':
                print('tool_start', event)
                progress_text = '思考中..., 考虑执行工具调用...'
                progress_bar = st.progress(0, text=progress_text)

            if kind == 'on_tool_end':
                progress_text = f'''工具{event['name']}完成，开始整理结果'''
                progress_bar.progress(100, text=progress_text)
                time.sleep(0.5)
                progress_bar.empty()

            if kind == 'on_chat_model_stream':
                content = event['data']['chunk'].content
                if content:
                    assistant_answer.re_render_answer(content)

            if kind == 'on_chain_end' and event['name'] == 'LangGraph':
                if event['data']['output'].get('llm'):
                    message = event['data']['output']['llm']['messages'][-1].content
                    assistant_answer.confirm_answer(message)
                if event['data']['output'].get('messages'):
                    message = event['data']['output']['messages'][-1].content
                    assistant_answer.confirm_answer(message)
                return message
