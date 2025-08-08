"""MDP环境 - 移除重复代码"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import logging

from core import TOKEN_DEFINITIONS, TOKEN_TO_INDEX, INDEX_TO_TOKEN, TOTAL_TOKENS, TokenType, RPNValidator
from alpha import FormulaEvaluator

logger = logging.getLogger(__name__)


class MDPState:
    """MDP环境的状态"""

    def __init__(self):
        self.token_sequence = [TOKEN_DEFINITIONS['BEG']]
        self.step_count = 0
        self.stack_size = 0

    def add_token(self, token_name):
        token = TOKEN_DEFINITIONS[token_name]
        self.token_sequence.append(token)
        self.step_count += 1

        # 更新栈大小
        if token.type == TokenType.OPERAND:
            if not token.name.startswith('delta_'):
                self.stack_size += 1
        elif token.type == TokenType.OPERATOR:
            self.stack_size = self.stack_size - token.arity + 1

    def encode_for_network(self):
        """编码状态用于神经网络输入"""
        max_length = 30
        encoding = np.zeros((max_length, TOTAL_TOKENS + 3))

        for i, token in enumerate(self.token_sequence[:max_length]):
            if i >= max_length:
                break

            token_idx = TOKEN_TO_INDEX[token.name]
            encoding[i, token_idx] = 1
            encoding[i, TOTAL_TOKENS] = i / max_length
            encoding[i, TOTAL_TOKENS + 1] = self.stack_size / 10.0
            encoding[i, TOTAL_TOKENS + 2] = self.step_count / max_length

        return encoding

    def to_formula_string(self):
        """将Token序列转换为可读的公式字符串"""
        from ..core import RPNEvaluator
        return RPNEvaluator.tokens_to_infix(self.token_sequence)

    def copy(self):
        """深拷贝状态"""
        new_state = MDPState()
        new_state.token_sequence = self.token_sequence.copy()
        new_state.step_count = self.step_count
        new_state.stack_size = self.stack_size
        return new_state


class AlphaMiningMDP:
    """完整的马尔可夫决策过程环境"""

    def __init__(self):
        self.max_episode_length = 30
        self.current_state = None
        self.formula_evaluator = FormulaEvaluator()  # 使用统一的评估器

    def reset(self):
        """开始新的episode"""
        self.current_state = MDPState()
        return self.current_state

    def step(self, action_token):
        """执行一个动作"""
        if not self.is_valid_action(action_token):
            return self.current_state, -1.0, True

        self.current_state.add_token(action_token)

        if action_token == 'END':
            done = True
        else:
            done = False

        if self.current_state.step_count >= self.max_episode_length:
            done = True

        return self.current_state, 0.0, done

    def is_valid_action(self, action_token):
        """检查动作是否合法"""
        valid_actions = RPNValidator.get_valid_next_tokens(self.current_state.token_sequence)
        return action_token in valid_actions

    def get_valid_actions(self):
        """获取当前状态的合法动作"""
        return RPNValidator.get_valid_next_tokens(self.current_state.token_sequence)