import pandas as pd
import random
from enum import Enum
from typing import Literal, Tuple
from riskchat.core.chat_agent import get_prompt
from pathlib import Path
import os
_SRC_PATH = Path(__file__).resolve().parents[1]
_PROMPT = get_prompt(
    os.path.join(_SRC_PATH, 'collection', 'trainer_prompts.yaml')
)
EVALUATION_PROMPT = _PROMPT['EVALUATION_PROMPT']


class Scenario(Enum):
    EASY = 0
    MEDIUM = 1
    HARD = 2

    # 仅作为 system 的内容摘要展示，实际发给模型的是 content
    @property
    def intro(self) -> str:
        return {
            Scenario.EASY: TRAINER_SCENARIO_EASY_INTRO,
            Scenario.MEDIUM: TRAINER_SCENARIO_MEDIUM_INTRO,
            Scenario.HARD: TRAINER_SCENARIO_HARD_INTRO
        }[self]

    @property
    def content_list(self) -> list:
        return {
            Scenario.EASY: TRAINER_SCENARIO_EASY_CONTENT_LIST,
            Scenario.MEDIUM: TRAINER_SCENARIO_MEDIUM_CONTENT_LIST,
            Scenario.HARD: TRAINER_SCENARIO_HARD_CONTENT_LIST
        }[self]


TRAINER_SCENARIO_EASY_INTRO = _PROMPT['TRAINER_SCENARIO_EASY_INTRO']
TRAINER_SCENARIO_MEDIUM_INTRO = _PROMPT['TRAINER_SCENARIO_MEDIUM_INTRO']
TRAINER_SCENARIO_HARD_INTRO = _PROMPT['TRAINER_SCENARIO_HARD_INTRO']
TRAINER_SCENARIO_EASY_CONTENT_LIST = _PROMPT['TRAINER_SCENARIO_EASY_CONTENT_LIST'].split('\n')
TRAINER_SCENARIO_MEDIUM_CONTENT_LIST = _PROMPT['TRAINER_SCENARIO_MEDIUM_CONTENT_LIST'].split('\n')
TRAINER_SCENARIO_HARD_CONTENT_LIST = _PROMPT['TRAINER_SCENARIO_HARD_CONTENT_LIST'].split('\n')

if __name__ == "__main__":
    # select_scenario('初级')
    pass
