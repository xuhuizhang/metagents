from riskchat.core.rag_agent import RagLLM
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser


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
