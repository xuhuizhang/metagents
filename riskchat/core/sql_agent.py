"""
Author: Hugo
Description: for calling sql
"""

import os
from operator import itemgetter
import pandas as pd
from sqlalchemy.engine.create import create_engine
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from riskchat import _SRC_PATH
from riskchat.core.chat_agent import get_prompt


class ConnectSQL:
    """
    connect to the db, return different types of engine
    """
    _DB_USER = 'app_risk'
    _DB_PASSWORD = 'uiwe8_k34kJSKJdwd'
    _DB_HOST = 'wqdcsrv3396.cn.infra:3313'
    _DB_NAME = 'db_risk'

    def __init__(self):
        pass

    # load the db from langchain
    def db_for_chain(self):
        return SQLDatabase.from_uri(
            f"mysql+pymysql://{self._DB_USER}:{self._DB_PASSWORD}@{self._DB_HOST}/{self._DB_NAME}"
        )

    def get_db_info(self):
        print(self.db_for_chain().get_table_info())

    # create a sql engine for pandas
    def db_sqlalchemy(self):
        return create_engine(f"mysql+pymysql://{self._DB_USER}:{self._DB_PASSWORD}@{self._DB_HOST}/{self._DB_NAME}?charset=utf8")


class GetSQLChain(ConnectSQL):
    """
    langchain 的sql agent
    """
    _AGENT_PROMPT: str = get_prompt(os.path.join(
        _SRC_PATH, 'libs', 'prompts.yaml'))['sql_prompt']

    def __init__(self, model, user_question: str, table_info: str):
        super().__init__()
        self.model = model
        self.user_question = user_question
        self.table_info = table_info

    @staticmethod
    def write_query_split(x: str):
        return x.split("```sql")[1].split("```")[0]

    # def get_schema(self):
    #     return super().db_for_chain().get_table_info()

    def run_query(self, query):
        return super().db_for_chain().run(query)

    # return sql scripts
    def sql_scripts(self) -> str:
        """
        :return: sql scripts
        """
        prompt = ChatPromptTemplate.from_template(self._AGENT_PROMPT)
        chain = (
            {
                'database': itemgetter('database'), 'question': itemgetter('question')
            }
            | prompt
            | self.model
            | StrOutputParser()
            | RunnableLambda(self.write_query_split)
        )
        response = chain.invoke(
            {
                'question': self.user_question, 'database': self.table_info
            }
        )
        return response

    # return the final SQL result
    # def sql_results(self):
    #     script = self.sql_scripts()
    #     return self.run_query(script)

    # def sql_to_pandas(self):
    #     sql = self.sql_scripts()
    #     print(f"sql代码如下 \n {sql}")
    #     if '%' in sql:
    #         sql = sql.replace('%', '%%')
    #     engine = super().db_sqlalchemy()
    #     df = pd.read_sql(sql=sql, con=engine)
    #     return sql, df
    # return the final SQL result
    def sql_results(self, scripts: str):
        return self.run_query(scripts)

    @staticmethod
    def sql_to_pandas(scripts: str) -> pd.DataFrame:
        if '%' in scripts:
            scripts = scripts.replace('%', '%%')
        print(f"sql代码如下 \n {scripts}")
        engine = ConnectSQL().db_sqlalchemy()
        try:
            df = pd.read_sql_query(scripts, engine)
        except Exception as e:
            print(f"读取数据时发生错误: {e}")
        return df


if __name__ == '__main__':
    sql_agent = ConnectSQL()
    # print(sql_agent.get_db_info())
    df = GetSQLChain.sql_to_pandas(
        ''' 
        SELECT amt_income_main_occupation, flag_approved, name_education_en 
        FROM f_al_ai_staff_base_v2 
        ORDER BY amt_income_main_occupation DESC 
        LIMIT 10
    '''
    )
    print(df)
