from riskchat import RagLLM, GetSQLChain, ask_llm, GetPythonChain, LLMList
from riskchat import _RAG_FILE_PATH, _BACKGROUND_DOC, _DB_SCHEMA_DOC

ds_chat = LLMList('hcc_deepseek_chat').chat_model(1)


class RiskWorkflow:

    _WORKFLOW_MATERIAL_PATH = _RAG_FILE_PATH
    _SELF_CHECK = '''
    根据我提供的内容，查询一下前后逻辑是否正确。
    已知内容：{topic}
    '''
    _WORKFLOW_PROMPT = [
        ''' 
        角色：专业的金融风控分析师。
        任务：基于问题描述与业务背景，详尽列举解决该问题所需的全部数据库字段及每字段的含义。如果问题涉及多个子任务，还需识别每一中间过程所需的数据库字段及其解释。
        问题描述：{question}
        业务背景：{background}
        操作指南：
        1. 深入分析问题与业务背景，构思解决问题的逻辑路径。
        2. 从问题描述中直接识别并提取所有相关的数据库字段。
        3. 若问题复杂，分解为多个子任务，并针对每个子任务分别识别所需的数据库字段。
        4. 对于每一个识别到的字段，根据问题描述和业务背景提供简明解释，说明其在解决当前问题中的作用。
        5. 整理所有识别到的字段和解释，包括问题直接关联的字段和各子任务中涉及的字段。
        6. 将这些字段以Python列表的形式返回，确保列表中的字段名称清晰无误。

        示例输出格式：Python代码\n variable = ['字段1，解释1', '字段2，解释2', ..., '字段N，解释N'] \n

        注意：对于每一个子任务或步骤，都应细致检查其对数据库字段的需求，以确保最终列表的全面性和准确性。
        ''',
        ''' 
        角色：精准洞察力与专业分析技巧兼备的金融风控分析师。
        任务： 基于详实的问题描述、业务代码、背景信息及数据集，撰写一份结构严密、逻辑连贯的风控分析报告。本报告旨在深入剖析业务风险，提炼关键洞见，并为决策层提供有力的数据支持与策略建议。
        报告要求：
        包括不限于以下的分析内容：
        1. 对于问题逻辑深度解析：
            * 详述分析问题的逻辑框架
            * 探索其他潜在的分析维度和方向。
        2. 对于最终数据的结论分析：
            * 数据的描述性分析：数据的趋势情况，数据间的异同点，异常情况等等
            * 数据之间的对比分析：包括同维度不同时间点的对比，不同维度相同时间点的对比等等
            * 数据的因果分析：通过数据之间的关联性，分析数据之间可能的因果关系
            * 提出问题：检查数据之间的可疑情况，提出相关需要更进一步分析的问题     
        3. 综合分析：基于上述分析，结合风险管理，市场推广等角度给出业务建议。
        
        问题描述：{question}
        业务背景：{background}
        业务代码：{code}
        业务数据：{result}
        '''
    ]
    # _RISK_AR_CHECK = [
    #     '''
    #     已知信息: {context}
    #     问题: {question}
    #     根据已知信息，提取问题中所需要的变量名和对应的描述。
    #     如果无法从中得到答案，请说“没有提供足够的相关信息”，不允许在答案中添加编造成分。
    #     '''
    # ]

    def __init__(self):
        pass

    def data_explore(self, user_question):
        rag_chain_00 = RagLLM(_BACKGROUND_DOC['UW_01'], 1000, 0, 10)
        retriever_00 = rag_chain_00.rerank_retriever_docs(
            user_question)
        retriever_00_str = f'\n{"*"*10}\n'.join(retriever_00)
        print('+'*10, '下面分析一下可能需要的业务逻辑\n')
        print(retriever_00_str)
        ans_01 = ask_llm(
            model=ds_chat, prompt_template=self._WORKFLOW_PROMPT[
                0], question=user_question, background=retriever_00_str
        )
        print(ans_01)
        ans_01_scripts = GetPythonChain.write_python_split(ans_01)
        ans_01_attr_list = GetPythonChain.python_exec(ans_01_scripts)
        ans_01_docs: str = ''
        print('+'*10, '下面分析一下可能需要的数据库字段\n')
        for ans_01_attr in ans_01_attr_list:
            rag_chain_01 = RagLLM(_DB_SCHEMA_DOC['UW_01'], 1000, 0, 3, 1)
            retriever_01 = rag_chain_01.rerank_retriever_docs(
                ans_01_attr)
            retriever_01_str = ''.join(retriever_01)
            print(retriever_01_str)
            ans_01_docs = ans_01_docs + '\n' + retriever_01_str
        sql_question_02 = user_question + '\n' + retriever_00_str
        ans_02 = GetSQLChain(ds_chat, sql_question_02, ans_01_docs)
        sql_code = ans_02.sql_scripts()
        ans_02_df = ans_02.sql_to_pandas(sql_code)
        print('+'*10, '最终数据结果如下\n')
        print(ans_02_df)
        ans_02_df_js = ans_02_df.to_json(orient='records')
        report = ask_llm(
            model=ds_chat, prompt_template=self._WORKFLOW_PROMPT[
                1], question=user_question, background=retriever_00, code=sql_code, result=ans_02_df_js
        )
        print('+'*10, '业务分析思路和数据报告分析\n')
        print(report)
        return ans_02_df


if __name__ == '__main__':
    question = '计算2BOD的合同总数'
    # model_llm = LLMList('deepseek_chat').chat_model(0.1)
    # model_llm.invoke('question')
    answer = RiskWorkflow()
    response = answer.data_explore(question)
    print(response)
