""" 
Author: Hugo
Description: model list for selection
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import (
    HuggingFaceBgeEmbeddings,
    XinferenceEmbeddings,
)
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from xinference.client import Client
from typing import Callable
from riskchat import _EMBEDDING_MODEL_PATH, _RERANKER_MODEL_PATH

load_dotenv(override=True)


class LLMList:
    """
    model selection, return a runnable model, please use model.invoke
    """

    def __init__(self, model_name: str):
        """
        根据跟定的模型名字。 初始化大模型的参数。

        :参数 model_info: api调用时候的大模型名字。
        """
        self.model_name: str = model_name
        self.model_info: str = self._get_model_info()['model_info']
        self._xinference = 'http://10.26.9.148:9997'
        os.environ['OPENAI_API_KEY'], os.environ['OPENAI_API_BASE'] = (
            self._get_api_info()
        )

    def _get_model_info(self) -> dict:
        _model_schema = {
            'hcc_deepseek_chat': {'model_info': 'ds/deepseek-chat', 'index': 0},
            'hcc_qwen2_sf72b': {'model_info': 'sf/Qwen2-72B-Instruct', 'index': 0},
            'hcc_local_qwen2': {'model_info': 'qwen2-instruct', 'index': 0},
            'hcc_bge_m3': {'model_info': 'bge-m3', 'index': 0},
            'hcc_bge_reranker': {'model_info': 'bge-reranker-large', 'index': 0},
            'local_bge_m3': {'model_info': _EMBEDDING_MODEL_PATH, 'index': 0},
            'local_bge_reranker': {'model_info': _RERANKER_MODEL_PATH, 'index': 0},
            'spark_pro': {'model_info': 'generalv3', 'index': 1},
            'spark_ultra': {'model_info': '4.0Ultra', 'index': 1},
            'doubao_func': {'model_info': 'ep-20240707133049-5vr4b', 'index': 2},
            'doubao_pro': {'model_info': 'ep-20240719164928-jhlrv', 'index': 2},
            'chat_gpt': {'model_info': 'gpt-3.5-turbo', 'index': 3},
            'deepseek_chat': {'model_info': 'deepseek-chat', 'index': 4},
            'glm4': {'model_info': 'glm-4', 'index': 5},
            'glm4_0520': {'model_info': 'glm-4-0520', 'index': 5},
            'glm4_p': {'model_info': 'glm-4-plus', 'index': 5},
            'glm4_alltools': {'model_info': 'glm-4-alltools', 'index': 5},
        }
        return _model_schema[self.model_name]

    def _get_api_info(self) -> tuple[str, str]:
        """get api key"""
        model_info = self._get_model_info()
        if os.getenv('API_KEY', ''):
            _api_key_list = os.getenv('API_KEY', '').split(',')
            _api_base_list = os.getenv('API_BASE', '').split(',')
        else:
            _api_key_list = [
                'sk-dE8NUvYDaXNHQhY284C6B0Dd454f4313Ab4fFeF962Ea8241']
            _api_base_list = ['http://10.26.9.148:3007/v1']

        _api_key = _api_key_list[model_info['index']].strip()
        _api_base = _api_base_list[model_info['index']].strip()
        return _api_key, _api_base

    def chat_model(self, temperature=0.1, streaming=False) -> ChatOpenAI:
        """chat model"""
        return ChatOpenAI(model=self.model_info, temperature=temperature, streaming=streaming)

    def openai_embedding(self):
        """embedding model"""
        return OpenAIEmbeddings(model=self.model_info)

    def embedding(self):
        """embedding model"""
        return XinferenceEmbeddings(
            server_url=self._xinference, model_uid=self.model_info
        )

    def local_embedding(self):
        """local embedding model"""
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_info,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )
        return embeddings

    def reranker(self, docs: list, question: str):
        rerank_model = Client(self._xinference).get_model(self.model_info)

        return rerank_model.rerank(  # type: ignore
            documents=docs, query=question, return_documents=True
        )

    def local_reranker(self):
        return HuggingFaceCrossEncoder(model_name=self.model_info)


# testing code
def reranker_test():
    query = 'A man is eating pasta.'
    docs = ['A man is eating food.', 'A man is eating a piece of bread.']
    rank = LLMList('hcc_bge_reranker').reranker(docs, query)
    print(rank)


def embedding_test():
    query = ['分析一下2BOD中HIGHAR和非HIGHAR不同个两组']
    rag = LLMList('hcc_bge_m3').embedding()
    print(len(rag.embed_documents(query)))
    print(rag.embed_documents(query))


def local_embedding_test():
    query = ['分析一下2BOD中HIGHAR和非HIGHAR不同个两组']
    rag = LLMList('local_bge_m3').local_embedding()
    print(len(rag.embed_documents(query)))
    print(rag.embed_documents(query))


def chat_test():
    model = LLMList('doubao_func').chat_model(1.0)
    response = model.invoke('你好')
    print(response)


if __name__ == '__main__':
    print('local', '*' * 30)
    local_embedding_test()
    print('one_api', '*' * 30)
    embedding_test()
    chat_test()
