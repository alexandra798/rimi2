"""统一的Alpha公式评估器 - 重构版"""
import pandas as pd
import numpy as np
import logging
import signal
from contextlib import contextmanager
from functools import lru_cache
from typing import Union, Dict, Optional, Any

from core import RPNEvaluator, RPNValidator, TOKEN_DEFINITIONS, TokenType, Operators

logger = logging.getLogger(__name__)


class FormulaEvaluator:
    """统一的RPN公式评估器 - 所有公式都作为RPN处理"""

    def __init__(self):

        self.rpn_evaluator = RPNEvaluator
        self.operators = Operators
        self._result_cache = {}  # 结果缓存

    def evaluate(self, formula: str, data: Union[pd.DataFrame, Dict],
                 allow_partial: bool = False) -> pd.Series:
        """
        统一的评估接口 - 所有公式都作为RPN处理

        Args:
            formula: RPN公式字符串
            data: 数据（DataFrame或字典）
            allow_partial: 是否允许部分表达式
            use_cache: 是否使用缓存

        Returns:
            评估结果的Series，失败时返回NaN Series
        """
        # 生成缓存键

        cache_key = self._generate_cache_key(formula, data, allow_partial)
        if cache_key in self._result_cache:
            logger.debug(f"Cache hit for formula: {formula[:50]}...")
            return self._result_cache[cache_key].copy()

        # 执行评估
        try:
            result = self._evaluate_impl(formula, data, allow_partial)
            # 缓存结果
            if result is not None:
                self._result_cache[cache_key] = result.copy()
            return result

        except TimeoutError as e:
            logger.warning(f"Formula evaluation timed out: {formula[:50]}...")
            return self._create_nan_series(data)
        except Exception as e:
            logger.error(f"Error evaluating formula '{formula[:50]}...': {str(e)}")
            logger.debug(f"Exception type: {type(e).__name__}", exc_info=True)
            return self._create_nan_series(data)

    def _evaluate_impl(self, formula: str, data: Union[pd.DataFrame, Dict],
                       allow_partial: bool) -> pd.Series:
        """实际的评估实现"""
        # 解析Token序列
        token_sequence = self._parse_tokens(formula)
        if not token_sequence:
            logger.warning(f"Failed to parse formula: {formula[:50]}...")
            return self._create_nan_series(data)

        # 验证Token序列
        if not allow_partial and not self._is_complete_expression(token_sequence):
            logger.warning(f"Incomplete RPN expression: {formula[:50]}...")
            if not RPNValidator.is_valid_partial_expression(token_sequence):
                return self._create_nan_series(data)

        # 准备数据
        data_dict = self._prepare_data(data)
        if data_dict is None:
            return self._create_nan_series(data)

        # 评估RPN表达式
        try:
            result = self.rpn_evaluator.evaluate(
                token_sequence,
                data_dict,
                allow_partial=allow_partial
            )

            # 转换结果为Series
            return self._convert_to_series(result, data)

        except Exception as e:
            logger.error(f"RPN evaluation failed: {str(e)}")
            return self._create_nan_series(data)

    def _parse_tokens(self, formula: str) -> list:
        """
        解析公式字符串为Token序列
        Args:
            formula: RPN公式字符串
        Returns:
            Token序列列表，解析失败返回空列表
        """
        try:
            token_names = formula.strip().split()
            token_sequence = []

            for name in token_names:
                if name in TOKEN_DEFINITIONS:
                    token_sequence.append(TOKEN_DEFINITIONS[name])
                else:
                    # 尝试解析动态常数（如const_3.14）
                    if name.startswith('const_'):
                        try:
                            value = float(name[6:])  # 去掉'const_'前缀
                            # 创建动态Token
                            from core.token_system import Token, TokenType
                            dynamic_token = Token(TokenType.OPERAND, name, value=value)
                            token_sequence.append(dynamic_token)
                        except ValueError:
                            logger.warning(f"Unknown token: {name}")
                            return []
                    else:
                        logger.warning(f"Unknown token: {name}")
                        return []

            return token_sequence

        except Exception as e:
            logger.error(f"Token parsing error: {e}")
            return []

    def _prepare_data(self, data: Union[pd.DataFrame, Dict]) -> Optional[Dict]:
        """准备数据为字典格式 - 确保都是 Series"""
        try:
            if isinstance(data, pd.DataFrame):
                # DataFrame转字典，使用Series格式
                return data.to_dict('series')
            elif isinstance(data, dict):
                prepared = {}
                # 获取一个参考索引
                ref_index = None
                for value in data.values():
                    if isinstance(value, pd.Series):
                        ref_index = value.index
                        break

                for key, value in data.items():
                    if isinstance(value, pd.Series):
                        prepared[key] = value
                    elif isinstance(value, np.ndarray):
                        # 转换为 Series
                        prepared[key] = pd.Series(value, index=ref_index)
                    else:
                        # 转换为 Series
                        prepared[key] = pd.Series(value, index=ref_index)
                return prepared
            else:
                logger.error(f"Unsupported data type: {type(data)}")
                return None
        except Exception as e:
            logger.error(f"Data preparation error: {e}")
            return None

    def _convert_to_series(self, result: Any, original_data: Union[pd.DataFrame, Dict]) -> pd.Series:
        """
        将评估结果转换为Series

        Args:
            result: 评估结果
            original_data: 原始数据（用于获取索引）

        Returns:
            结果Series
        """
        try:
            # 如果已经是Series，直接返回
            if isinstance(result, pd.Series):
                return result

            # 获取索引
            if isinstance(original_data, pd.DataFrame):
                index = original_data.index
            elif isinstance(original_data, dict):
                # 从字典中找一个Series获取索引
                for value in original_data.values():
                    if isinstance(value, pd.Series):
                        index = value.index
                        break
                else:
                    index = None
            else:
                index = None

            # 转换为Series
            if isinstance(result, np.ndarray):
                return pd.Series(result, index=index)
            elif isinstance(result, (int, float, np.number)):
                if index is not None:
                    return pd.Series(result, index=index)
                else:
                    return pd.Series([result])
            else:
                # 尝试直接转换
                return pd.Series(result, index=index)

        except Exception as e:
            logger.error(f"Result conversion error: {e}")
            return self._create_nan_series(original_data)

    def _create_nan_series(self, data: Union[pd.DataFrame, Dict]) -> pd.Series:
        """创建NaN Series作为错误返回值"""
        if isinstance(data, pd.DataFrame):
            return pd.Series(np.nan, index=data.index)
        elif isinstance(data, dict):
            # 尝试从字典中找一个Series获取长度和索引
            for value in data.values():
                if isinstance(value, pd.Series):
                    return pd.Series(np.nan, index=value.index)
                elif isinstance(value, np.ndarray):
                    return pd.Series(np.nan, index=range(len(value)))
        return pd.Series(np.nan)

    def _is_complete_expression(self, token_sequence: list) -> bool:
        """检查是否为完整的RPN表达式"""
        if not token_sequence:
            return False

        # 必须以BEG开始
        if token_sequence[0].name != 'BEG':
            return False

        # 可以以END结束（完整）或不以END结束（部分）
        has_end = token_sequence[-1].name == 'END' if len(token_sequence) > 1 else False

        # 验证栈平衡
        stack_size = RPNValidator.calculate_stack_size(token_sequence)

        if has_end:
            return stack_size == 1  # 完整表达式应该留下1个结果
        else:
            return stack_size >= 1  # 部分表达式至少有1个元素

    def _generate_cache_key(self, formula: str, data: Any, allow_partial: bool) -> str:
        """生成缓存键"""
        # 使用公式和数据ID以及allow_partial标志作为键
        data_id = id(data)
        return f"{formula}_{data_id}_{allow_partial}"

    def evaluate_state(self, state, X_data) -> Optional[np.ndarray]:
        """
        评估状态对应的公式值

        Args:
            state: MDPState对象
            X_data: 数据

        Returns:
            评估结果的数组，失败返回None
        """
        try:
            # 构建RPN字符串
            rpn_string = ' '.join([t.name for t in state.token_sequence])

            # 评估
            result = self.evaluate(rpn_string, X_data, allow_partial=True)

            # 转换为数组
            if result is not None:
                if hasattr(result, 'values'):
                    return result.values
                else:
                    return np.array(result)
            return None

        except Exception as e:
            logger.error(f"Error evaluating state: {e}")
            return None





