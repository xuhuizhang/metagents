from langchain_community.document_loaders import CSVLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.document_transformers import LongContextReorder
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter


def csv_parser(file_path):
    loader = CSVLoader(
        file_path, encoding='utf-8', csv_args={'delimiter': ','}
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(separators=['\t'])
    splitted_documents = text_splitter.split_documents(documents)
    return splitted_documents
