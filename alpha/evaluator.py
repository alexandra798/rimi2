"""统一的Alpha公式评估器"""
import pandas as pd
import numpy as np
import logging
import re
from functools import lru_cache
from typing import Union, Dict, Optional

from core import RPNEvaluator, RPNValidator, TOKEN_DEFINITIONS, Operators

logger = logging.getLogger(__name__)


class FormulaEvaluator:
    """统一的公式评估接口 - 支持RPN和传统格式"""

    def __init__(self):
        self.rpn_evaluator = RPNEvaluator
        self.operators = Operators
        self._cache = {}

    def evaluate(self, formula: str, data: Union[pd.DataFrame, Dict],
                 allow_partial: bool = False) -> pd.Series:
        """
        统一的评估接口

        Args:
            formula: 公式字符串（RPN或传统格式）
            data: 数据（DataFrame或字典）
            allow_partial: 是否允许部分表达式

        Returns:
            评估结果的Series
        """
        # 检查缓存
        cache_key = (formula, id(data), allow_partial)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 判断公式类型
        if self.is_rpn_formula(formula):
            result = self.evaluate_rpn(formula, data, allow_partial)
        else:
            result = self.evaluate_traditional(formula, data)

        # 缓存结果
        self._cache[cache_key] = result

        return result

    def is_rpn_formula(self, formula: str) -> bool:
        """判断是否为RPN格式的公式"""
        rpn_indicators = ['BEG', 'END', 'add', 'sub', 'mul', 'div',
                          'ts_mean', 'ts_std', 'delta_']
        return any(indicator in formula for indicator in rpn_indicators)

    def evaluate_rpn(self, formula: str, data: Union[pd.DataFrame, Dict],
                     allow_partial: bool = False) -> pd.Series:
        """评估RPN格式的公式"""
        try:
            # 解析RPN字符串为Token序列
            token_names = formula.split()
            token_sequence = []

            for name in token_names:
                if name in TOKEN_DEFINITIONS:
                    token_sequence.append(TOKEN_DEFINITIONS[name])
                else:
                    logger.warning(f"Unknown token in RPN formula: {name}")
                    if isinstance(data, pd.DataFrame):
                        return pd.Series(np.nan, index=data.index)
                    else:
                        return pd.Series(np.nan)

            # 转换数据为字典格式
            if isinstance(data, pd.DataFrame):
                data_dict = data.to_dict('series')
            else:
                data_dict = data

            # 判断是否为完整表达式
            is_complete = (len(token_sequence) > 0 and
                           token_sequence[-1].name == 'END')

            # 使用RPN求值器评估
            result = self.rpn_evaluator.evaluate(
                token_sequence,
                data_dict,
                allow_partial=allow_partial or not is_complete
            )

            # 确保返回Series
            if result is not None:
                if isinstance(result, pd.Series):
                    return result
                else:
                    if isinstance(data, pd.DataFrame):
                        return pd.Series(result, index=data.index)
                    else:
                        return pd.Series(result)
            else:
                if isinstance(data, pd.DataFrame):
                    return pd.Series(np.nan, index=data.index)
                else:
                    return pd.Series(np.nan)

        except Exception as e:
            logger.error(f"Error evaluating RPN formula '{formula}': {e}")
            if isinstance(data, pd.DataFrame):
                return pd.Series(np.nan, index=data.index)
            else:
                return pd.Series(np.nan)

    def evaluate_traditional(self, formula: str, data: pd.DataFrame) -> pd.Series:
        """评估传统格式的公式"""
        try:
            # 清理和验证公式
            sanitized_formula = self._sanitize_formula(formula)
            if sanitized_formula is None:
                logger.error(f"Formula failed security validation: '{formula}'")
                return pd.Series(np.nan, index=data.index)

            # 创建安全的评估环境
            safe_dict = {}

            # 添加数据列
            for col in data.columns:
                safe_dict[col] = data[col]

            # 添加允许的函数
            allowed_functions = {
                'abs': abs,
                'max': max,
                'min': min,
                'sum': sum,
                'len': len,
                'safe_divide': self.operators.safe_divide,
                'ts_ref': self.operators.ts_ref,
                'csrank': self.operators.csrank,
                'sign': self.operators.sign,
                'abs_op': self.operators.abs_op,
                'log': self.operators.log,
                'greater': self.operators.greater,
                'less': self.operators.less,
                'ts_rank': self.operators.ts_rank,
                'std': self.operators.std,
                'ts_max': self.operators.ts_max,
                'ts_min': self.operators.ts_min,
                'skew': self.operators.skew,
                'kurt': self.operators.kurt,
                'mean': self.operators.mean,
                'med': self.operators.med,
                'ts_sum': self.operators.ts_sum,
                'cov': self.operators.cov,
                'corr': self.operators.corr,
                'decay_linear': self.operators.decay_linear,
                'wma': self.operators.wma,
                'ema': self.operators.ema,
                'np': np,
                'pd': pd,
            }

            # 验证公式中的变量
            formula_vars = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', formula)
            for var in formula_vars:
                if var not in safe_dict and var not in allowed_functions:
                    logger.warning(f"Unknown variable '{var}' in formula: {formula}")
                    return pd.Series(np.nan, index=data.index)

            safe_dict.update(allowed_functions)

            # 评估公式
            try:
                result = pd.eval(sanitized_formula, local_dict=safe_dict, engine='python')

                # 处理结果类型
                if isinstance(result, (int, float, np.number)):
                    result = pd.Series(result, index=data.index)
                elif isinstance(result, np.ndarray):
                    result = pd.Series(result, index=data.index)
                elif not isinstance(result, pd.Series):
                    try:
                        result = pd.Series(result, index=data.index)
                    except:
                        logger.error(f"Cannot convert result to Series for formula: {formula}")
                        return pd.Series(np.nan, index=data.index)

                # 替换无限值
                result = result.replace([np.inf, -np.inf], np.nan)

                return result

            except Exception as e:
                # 尝试使用标准eval
                if "scalar" in str(e).lower():
                    try:
                        eval_dict = {"__builtins__": {}}
                        eval_dict.update(safe_dict)

                        result = eval(sanitized_formula, eval_dict)

                        if isinstance(result, pd.Series):
                            return result.replace([np.inf, -np.inf], np.nan)
                        else:
                            return pd.Series(result, index=data.index).replace([np.inf, -np.inf], np.nan)

                    except Exception as eval_error:
                        logger.error(f"Both pd.eval and eval failed for formula '{formula}': {eval_error}")
                        return pd.Series(np.nan, index=data.index)
                else:
                    raise e

        except Exception as e:
            logger.error(f"Error evaluating formula '{formula}': {e}")
            return pd.Series(np.nan, index=data.index)

    def _sanitize_formula(self, formula: str) -> Optional[str]:
        """清理和验证公式安全性"""
        # 检查不安全的操作
        unsafe_patterns = [
            r'__\w+__',
            r'import\s+',
            r'exec\s*\(',
            r'eval\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\(',
        ]

        for pattern in unsafe_patterns:
            if re.search(pattern, formula, re.IGNORECASE):
                logger.warning(f"Unsafe formula detected: {formula}")
                return None

        # 验证只包含允许的字符
        allowed_chars = re.compile(r'^[a-zA-Z0-9_\s\+\-\*\/\(\)\.\,]+$')
        if not allowed_chars.match(formula):
            logger.warning(f"Formula contains invalid characters: {formula}")
            return None

        return formula

    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()

    def evaluate_batch(self, formulas: list, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        批量评估多个公式

        Args:
            formulas: 公式列表
            data: 数据

        Returns:
            公式到结果的字典
        """
        results = {}

        # 预处理共享数据
        data_dict = data.to_dict('series') if isinstance(data, pd.DataFrame) else data

        for formula in formulas:
            results[formula] = self.evaluate(formula, data)

        return results

    def evaluate_state(self, state, X_data):
        """评估状态对应的公式值 - 统一接口"""
        try:
            # 构建RPN字符串
            rpn_string = ' '.join([t.name for t in state.token_sequence])

            # 使用统一的evaluate方法
            result = self.evaluate(rpn_string, X_data, allow_partial=True)

            if result is not None:
                if hasattr(result, 'values'):
                    return result.values
                else:
                    return np.array(result)
            return None

        except Exception as e:
            logger.error(f"Error evaluating state: {e}")
            return None

    def evaluate_state_batch(self, states, X_data):
        """批量评估多个状态 - 减少重复计算"""
        results = []

        # 如果数据太大，先采样
        if len(X_data) > 10000:
            sample_indices = np.random.choice(len(X_data), 10000, replace=False)
            X_sample = X_data.iloc[sample_indices]
        else:
            X_sample = X_data

        for state in states:
            rpn_string = ' '.join([t.name for t in state.token_sequence])
            result = self.evaluate(rpn_string, X_sample, allow_partial=True)
            results.append(result)

        return results