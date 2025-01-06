"""
Author: Hugo
Description: for calling python agent
"""

import io
import os
from typing import Optional, Callable, Any
import pandas as pd
from operator import itemgetter
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from riskchat import _SRC_PATH
from riskchat.core.chat_agent import get_prompt


class PythonAgentError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class GetPythonChain:
    """
    base python agent
    """

    _AGENT_PROMPT: dict = get_prompt(
        os.path.join(_SRC_PATH, 'libs', 'prompts.yaml'))

    def __init__(
        self,
        model: Callable,
        question: str,
        dataframe: Optional[pd.DataFrame] = None,
        dataframe_name: Optional[str] = None,
    ):
        self.model = model
        self.question = question
        if dataframe_name:
            self.df = dataframe
            self.df_name = dataframe_name

    @staticmethod
    def write_python_split(x: str):
        return x.split('```python')[1].split('```')[0]

    # def get_table_info(self):
    #     if not isinstance(self.df, pd.DataFrame):
    #         raise TypeError('self.df must be a pandas DataFrame')
    #     buf = io.StringIO()
    #     self.df.info(buf=buf, memory_usage=False)
    #     df_info = f'{buf.getvalue()}\n数据的前五行如下：\n{self.df.head().to_string()}'
    #     return df_info

    def get_table_info(self):
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError('self.df must be a pandas DataFrame')
        df_shape_f = f'数据维度是：{self.df.shape[0]}行，{self.df.shape[1]}列'
        df_type = pd.DataFrame(
            {'Data Type': self.df.dtypes, 'Missing Ratio': self.df.isna().mean()}
        ).to_string()
        df_type_f = f'每一列的数据类型和缺失值的比例分别是：\n{df_type}'
        df_head_f = f'数据的前五行如下：\n{self.df.head().to_string()}'
        df_info = f'\n{df_shape_f}\n{"*"*30}\n{df_type_f}\n{"*"*30}\n{df_head_f}'
        return df_info

    def python_scripts(self):
        _input_dict = {
            'question': self.question,
            'info': self.get_table_info(),
            'name': self.df_name,
        }
        if self.df_name:
            _key_tuple = ('question', 'info', 'name')
            _template = self._AGENT_PROMPT['py_df_prompt']
        else:
            _key_tuple = 'question'
            _template = self._AGENT_PROMPT['py_prompt']

        _input = {key: _input_dict[key] for key in _key_tuple}
        prompt = ChatPromptTemplate.from_template(_template)
        chain = (
            {**{key: itemgetter(key) for key in _input.keys()}}
            | prompt
            | self.model
            | StrOutputParser()
            | RunnableLambda(self.write_python_split)
        )
        response = chain.invoke(_input)
        return response

    # generate the code by LLM
    # def python_scripts(self):
    #     if self.df_name:
    #         prompt = ChatPromptTemplate.from_template(self._AGENT_PROMPT['py_df_prompt'])
    #         chain = (
    #             {
    #                 'question': itemgetter('question'),
    #                 'info': itemgetter('info'),
    #                 'name': itemgetter('name')
    #             }
    #             | prompt
    #             | self.model
    #             | StrOutputParser()
    #             | RunnableLambda(self.write_python_split)
    #         )
    #         response = chain.invoke(
    #             {
    #                 'question': self.question,
    #                 'info': self.get_table_info(),
    #                 'name': self.df_name
    #             }
    #         )
    #     else:
    #         prompt = ChatPromptTemplate.from_template(self._AGENT_PROMPT['py_prompt'])
    #         chain = (
    #             {'question': itemgetter('question')}
    #             | prompt
    #             | self.model
    #             | StrOutputParser()
    #             | RunnableLambda(self.write_python_split)
    #         )
    #         response = chain.invoke({'question': self.question})
    #     # print(f'Python 的代码是：\n {response}')
    #     return response

    # run the code, and return the final result
    def python_results(self):
        py_scripts = self.python_scripts()
        print('执行如下python代码：\n', py_scripts)
        local_vars = {self.df_name: self.df}
        try:
            exec(py_scripts, local_vars)
            return local_vars['result']
        except Exception as e:
            raise PythonAgentError(f'执行代码报错:\n{py_scripts}\n报错内容: {str(e)}')

    @staticmethod
    def python_exec(scripts: str, df: Optional[pd.DataFrame] = None) -> Any:
        print('执行如下python代码：\n', scripts)
        if df is None:
            local_vars = {}
        else:
            local_vars = {'df': df}
        try:
            exec(scripts, local_vars)
            return next(reversed(local_vars.values()))
        except Exception as e:
            raise PythonAgentError(f'执行代码报错:\n{scripts}\n报错内容: {str(e)}')


def py_explore(model, user_question: str, df: Optional[pd.DataFrame], df_name: str):
    retries = 0
    while retries <= 3:
        try:
            ans = GetPythonChain(model, user_question, df, df_name)
            ans_df: Any = ans.python_results()
            print(f'尝试第 {retries+1} 次运行代码, 成功')
            return ans_df
        except Exception as e:
            print(f'尝试 {retries+1} 失败: {str(e)}')
            if retries < 3:
                user_question = f'{user_question} \n {str(e)}'
                retries += 1
            else:
                print('达到3次重试次数，停止运行。')
                return None


if __name__ == '__main__':
    print(GetPythonChain._AGENT_PROMPT['py_df_prompt'])
