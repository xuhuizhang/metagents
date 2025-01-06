''' 
agents for different chat use cases
'''

import yaml
from typing import Callable, Union, Dict, Any
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_prompt(prompt_path: str) -> dict:
    with open(prompt_path, 'r', encoding='utf-8') as file:
        prompt = yaml.safe_load(file)
    if not isinstance(prompt, dict):
        raise TypeError('Expected a dictionary but got something else.')
    return prompt


def ask_llm(
        model: Callable
        , prompt_template: str
        , **kwargs
    ) -> str:
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = (
            {
                **{key: itemgetter(key) for key in kwargs.keys()}
            }
            | prompt
            | model
            | StrOutputParser()
    )
    response = chain.invoke({**kwargs})
    return response

if __name__ == '__main__':
    pass
    
    
