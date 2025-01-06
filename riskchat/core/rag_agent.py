"""
Author: Hugo
Description: For RAG LLM
"""

import os
import pandas as pd
from typing import Callable, Literal, Optional, Sequence
from operator import itemgetter
from itertools import zip_longest
from langchain_community.document_loaders import (
    # PyPDFLoader,
    # PDFPlumberLoader,
    UnstructuredMarkdownLoader,
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredPowerPointLoader
)
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.document_transformers import LongContextReorder
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from riskchat import (
    _RAG_FILE_PATH, _RAG_VECTOR_STORE
)
from riskchat.libs.model_list import LLMList


class RagLLM:
    """
    this is a class for RAG LLM, it will load the documents and use the model 
    to answer the questions.
    """

    _FILE_PATH = _RAG_FILE_PATH
    _VECTOR_STORE = _RAG_VECTOR_STORE

    def __init__(
        self,
        file,
        chunk_size: int,
        chunk_overlap: int,
        k_retriever: int,
        top_n: int = 5,
        reranker_model: str = 'bge_reranker',
        local_embedding: bool = False
    ):
        self.file = file
        self.file_name = file.split('.')[0]  # xxx.pdf
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # top k documents for retriever model, default 3, max 20, min 1.
        self.k_retriever = k_retriever
        self.top_n = top_n
        self.reranker_model = reranker_model
        self.local_embedding = local_embedding

    # load documents
    def load_documents(self):
        file_format_loader_map = {
            'docx': Docx2txtLoader,
            'md': UnstructuredMarkdownLoader,
            'pdf': PyMuPDFLoader,
            'txt': TextLoader,
            'csv': CSVLoader,
            'pptx': UnstructuredPowerPointLoader
        }

        loader_class = file_format_loader_map.get(self.file.split('.')[-1])
        if not loader_class:
            raise ValueError(f'无法解析该文件格式: {self.file}')

        file_path = f'{self._FILE_PATH}/{self.file}'

        if self.file.endswith('.csv'):
            # col_list = pd.read_csv(
            #     file_path, encoding='utf-8', delimiter=',', nrows=2
            # ).columns.to_list()
            loader = loader_class(
                file_path, encoding='utf-8', csv_args={'delimiter': ','}
            )
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\t'], chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            splitted_documents = text_splitter.split_documents(documents)

        elif self.file.endswith('.txt'):
            loader = loader_class(file_path, encoding='utf-8')
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n', '。', '，', '；', '！', '？', ' '],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False
            )
            documents = loader.load()
            splitted_documents = text_splitter.split_documents(documents)
        elif self.file.endswith('.md'):
            loader = loader_class(file_path, encoding='utf-8')
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
            ]
            text_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on
            )
            documents = loader.load()
            splitted_documents = text_splitter.split_text(documents)
        else:
            loader = loader_class(file_path)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            documents = loader.load()
            splitted_documents = text_splitter.split_documents(documents)

        return splitted_documents

    def embed_documents(self, embedding_model: str = 'bge_m3'):
        if self.local_embedding:
            embeddings = LLMList('local_'+embedding_model).local_embedding()
        else:
            embeddings = LLMList('hcc_'+embedding_model).embedding()
        chunks = self.load_documents()
        folder = f'{self._VECTOR_STORE}/{self.file_name}_{self.chunk_size}_{self.chunk_overlap}'
        if not os.path.exists(folder):
            print("create new vectorstore")
            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local(folder)
        else:
            print("load old vectorstore")
            vectorstore = FAISS.load_local(
                folder, embeddings=embeddings, allow_dangerous_deserialization=True
            )
        return vectorstore

    def retriever(self):
        return self.embed_documents().as_retriever(search_kwargs={'k': self.k_retriever})

    def retriever_docs(self, question: str) -> list:
        _response = self.retriever().invoke(question)
        if _response:
            _docs = [doc.page_content for doc in _response]
        return _docs

    def rerank_retriever_docs(
        self,
        question: str
    ) -> list:
        if self.local_embedding:
            model = LLMList('local_'+self.reranker_model).local_reranker()
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=CrossEncoderReranker(model=model, top_n=self.top_n), base_retriever=self.retriever()
            )
            _rerank_response = compression_retriever.invoke(question)
            _rerank_docs = [doc.page_content for doc in _rerank_response]
        else:
            _docs = self.retriever_docs(question)
            _rerank_response = LLMList(
                'hcc_'+self.reranker_model).reranker(_docs, question)
            if _rerank_response:
                _rerank_docs = [
                    doc['document']['text']
                    for doc in _rerank_response['results'][: self.top_n]
                ]
        return _rerank_docs

    @staticmethod
    def format_retriever(docs: list[Document], to_list=False) -> list | str | Literal['无相关信息！']:
        docs_string = [doc.page_content for doc in docs]
        # print('检索内容为:\n', '\n\n'.join(docs_string))
        if to_list:
            return docs_string
        else:
            return '\n\n'.join(docs_string) if docs_string else '无相关信息！'

    @staticmethod
    def string_retriever(docs: list) -> str:
        return '\n\n'.join(docs) if docs else '无相关信息！'

    @staticmethod
    def reorder_retriever(docs: list[Document]) -> Sequence[Document]:
        """
        不太相关的文档被放置在列表的中间，而更相关的文档则被置于开始和结束
        有利于大模型处理长文本
        """
        reorder = LongContextReorder().transform_documents(docs)
        return reorder

    def retriever_chain(self, model, prompt):
        retriever = (
            {
                'context': itemgetter('question') | self.retriever() | self.format_retriever, 'question': itemgetter('question')
            }
            | prompt  # type: ignore
            | model
            | StrOutputParser()
        )
        return retriever

    def rerank_retriever_chain(self, model, prompt):
        retriever = (
            {
                'context': (
                    itemgetter('question')
                    | RunnableLambda(self.rerank_retriever_docs)
                    | RunnableLambda(self.string_retriever)
                ), 'question': itemgetter('question')
            }
            | prompt  # type: ignore
            | model
            | StrOutputParser()
        )
        return retriever


def rag_rerank_doc(question, file, chunk_size, chunk_overlap, k_retriever, top_n):
    _rag_chain = RagLLM(file, chunk_size, chunk_overlap, k_retriever, top_n)
    _retriever = _rag_chain.rerank_retriever_docs(question)
    _answer = _rag_chain.string_retriever(_retriever)
    return _answer


def rag_rerank_query(model, file, chunk_size, chunk_overlap, k_retriever, top_n):
    query = RagLLM(file, chunk_size, chunk_overlap, k_retriever, top_n)
    template = """
    角色：你是一个专业性强的对话专家。
    任务：请使用以下source标签内的信息，准确专业的回答问题。
    注意事项：如果你不知道答案，需表明文件中未提供具体信息，不要试图编造答案。
    <source> {context} </source>
    问题: {question}
    回答:
    """
    prompt = ChatPromptTemplate.from_template(template=template)
    rag_chain = query.rerank_retriever_chain(model, prompt)
    return rag_chain


def rag_query_with_history(model, file, chunk_size, chunk_overlap, k_retriever):
    # load history vector store for chat history.
    retriever_hist = RagLLM('History_VectorStore.txt', 64, 16, 2)
    retriever_file = RagLLM(file, chunk_size, chunk_overlap, k_retriever)
    template = """
    角色：你是一个专业性强的对话专家。
    任务：请使用以下chat_history和source标签内的信息， 准确专业的回答问题。
    注意事项：如果你不知道答案，需表明文件中未提供具体信息，不要试图编造答案。
    <chat_history> {history} </chat_history>
    <source> {context} </source>
    问题: {question}
    回答:
    """
    prompt = ChatPromptTemplate.from_template(template=template)
    rag_chain = (
        {
            'history': itemgetter('question') | retriever_hist.retriever()
            | retriever_hist.format_retriever, 'context': itemgetter('question')
            | retriever_file.retriever() | retriever_file.format_retriever, 'question': itemgetter('question')
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return rag_chain


def rag_test():
    from riskchat import LLMList
    model = LLMList('deepseek_chat').chat_model(0.3)
    _rag_chain = rag_rerank_query(
        model, 'credit_scoring_procedure_md.md', 2048, 512, 10, 5
    )
    _question = '''
    评分模型开发的细节有哪些
    '''
    response = _rag_chain.invoke({'question': _question})
    print(response)


def one_api_test():

    rag_chain = RagLLM('CFCS_business_background.csv', 1000, 0, 5)
    question = '''
    分析一下2BOD中HIGHAR和非HIGHAR不同个两组，分别计算approve rate, 以及fpd30的rate
    '''
    docs = rag_chain.retriever().invoke(question)
    for x in docs:
        print(x)


def local_test():
    rag_chain = RagLLM('CFCS_business_background.csv', 1000, 0, 5, True)
    question = '''
    分析一下2BOD中HIGHAR和非HIGHAR不同个两组，分别计算approve rate, 以及fpd30的rate
    '''
    docs = rag_chain.retriever().invoke(question)
    for x in docs:
        print(x)


if __name__ == '__main__':
    # print('local', '+'*30)
    # local_test()
    # print('one_api', '+'*30)
    # one_api_test()
    rag_test()
