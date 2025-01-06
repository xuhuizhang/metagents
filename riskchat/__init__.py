import os, sys
from pathlib import Path

_PATH_FILE = Path(__file__).resolve()
_SRC_PATH = str(_PATH_FILE.parents[0])
_PROJECT_PATH = str(_PATH_FILE.parents[1])

if _PROJECT_PATH not in sys.path:
    sys.path.append(_PROJECT_PATH)
    
_EMBEDDING_MODEL_PATH = 'C:/Users/hugo.zhangTJ/.cache/modelscope/hub/Xorbits/bge-m3'
_RERANKER_MODEL_PATH = 'C:/Users/hugo.zhangTJ/.cache/modelscope/hub/Xorbits/bge-reranker-large'
_RAG_FILE_PATH = os.path.join(_PROJECT_PATH, 'docs', 'RAG_Materials')
_RAG_VECTOR_STORE = os.path.join(_PROJECT_PATH, 'docs', 'Vector_Store')

_BACKGROUND_DOC = {'UW_01': 'CFCS_business_background.csv'}
_DB_SCHEMA_DOC = {'UW_01': 'table_info_f_al_ai_staff_base.csv'}


from .core import (
    ask_llm,
    get_prompt,
    GetPythonChain,
    RagLLM,
    GetSQLChain,
    ConnectSQL,
    rag_rerank_query,
    rag_rerank_doc,
)
from .libs import LLMList

__all__ = [
    'ask_llm',
    'get_prompt',
    'GetPythonChain',
    'RagLLM',
    'GetSQLChain',
    'LLMList',
    'ConnectSQL',
    'rag_rerank_query',
    'rag_rerank_doc',
]

if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
