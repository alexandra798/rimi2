"""RPN表达式求值器"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy import stats
import logging
from core.token_system import TokenType, TOKEN_DEFINITIONS
from core.operators import Operators

logger = logging.getLogger(__name__)


class RPNEvaluator:
    """评估RPN表达式的值"""

    @staticmethod
    def evaluate(token_sequence, data_dict, allow_partial=True):
        """
        评估RPN表达式 - 修正版支持部分表达式

        Args:
            token_sequence: Token序列
            data_dict: 数据字典
            allow_partial: 是否允许部分表达式（栈中有多个元素）
        Returns:
            评估结果（Series或数组）
        """
        stack = []

        # 获取数据长度和索引
        data_length = None
        data_index = None
        for key, value in data_dict.items():
            if isinstance(value, (pd.Series, np.ndarray)):
                if isinstance(value, pd.Series):
                    data_length = len(value)
                    data_index = value.index
                else:
                    data_length = len(value)
                break

        i = 1  # 跳过BEG
        while i < len(token_sequence):
            token = token_sequence[i]

            if token.name == 'END':
                break

            if token.type == TokenType.OPERAND:
                # 处理操作数
                if token.name in data_dict:
                    stack.append(data_dict[token.name])
                elif token.name.startswith('const_'):
                    const_value = float(token.name.split('_')[1])
                    if data_length and data_index is not None:
                        stack.append(pd.Series(const_value, index=data_index))
                    elif data_length:
                        stack.append(np.full(data_length, const_value))
                    else:
                        stack.append(const_value)
                elif token.name.startswith('delta_'):
                    # delta不应该单独出现，跳过
                    pass

            elif token.type == TokenType.OPERATOR:
                if token.name.startswith('ts_'):
                    # 时序操作符特殊处理
                    if len(stack) < 1:
                        logger.error(f"Insufficient operands for {token.name}")
                        return None

                    data_operand = stack.pop()
                    window = 5  # 默认窗口

                    # 检查下一个token是否是delta
                    if i + 1 < len(token_sequence) and token_sequence[i + 1].name.startswith('delta_'):
                        delta_token = token_sequence[i + 1]
                        window = int(delta_token.name.split('_')[1])
                        i += 2  # 跳到delta后面

                        # 应用时序操作
                        result = RPNEvaluator.apply_time_series_op(
                            token.name, data_operand, window
                        )
                        stack.append(result)
                        continue  # 重要：跳过主循环的 i += 1
                    else:
                        # 没有delta，使用默认窗口
                        result = RPNEvaluator.apply_time_series_op(
                            token.name, data_operand, window
                        )
                        stack.append(result)

                elif token.arity == 1:
                    # 一元操作符
                    if len(stack) < 1:
                        logger.error(f"Insufficient operands for {token.name}")
                        return None
                    operand = stack.pop()
                    result = RPNEvaluator.apply_unary_op(
                        token.name, operand, data_length, data_index
                    )
                    stack.append(result)

                elif token.arity == 2:
                    # 二元操作符
                    if len(stack) < 2:
                        logger.error(f"Insufficient operands for {token.name}")
                        return None
                    operand2 = stack.pop()
                    operand1 = stack.pop()
                    result = RPNEvaluator.apply_binary_op(
                        token.name, operand1, operand2, data_length, data_index
                    )
                    stack.append(result)

                elif token.arity == 3:
                    # 三元操作符
                    if len(stack) < 3:
                        logger.error(f"Insufficient operands for {token.name}")
                        return None
                    operand3 = stack.pop()
                    operand2 = stack.pop()
                    operand1 = stack.pop()
                    result = RPNEvaluator.apply_ternary_op(
                        token.name, operand1, operand2, operand3
                    )
                    stack.append(result)

            i += 1

        # 返回结果 - 修复：支持部分表达式
        if len(stack) == 0:
            logger.error("Empty stack after evaluation")
            return None
        elif len(stack) == 1:
            # 完整表达式的正常情况
            result = stack[0]
            if isinstance(result, (int, float, np.number)):
                if data_length and data_index is not None:
                    return pd.Series(result, index=data_index)
                elif data_length:
                    return np.full(data_length, result)
            return result
        else:
            # 部分表达式的情况
            if allow_partial:
                # 对于部分表达式，返回栈顶元素或组合多个元素
                # 策略1：返回栈顶元素
                result = stack[-1]

                # 策略2：如果有多个元素，可以尝试组合它们
                # 例如，计算所有元素的平均值作为部分表达式的值
                if len(stack) > 1:
                    logger.debug(f"Partial expression with {len(stack)} stack elements")
                    # 尝试将所有栈元素平均（这是一种启发式方法）
                    try:
                        # 确保所有元素都是相同形状的数组
                        arrays = []
                        for elem in stack:
                            if isinstance(elem, pd.Series):
                                arrays.append(elem.values)
                            elif isinstance(elem, np.ndarray):
                                arrays.append(elem)
                            else:
                                # 标量，扩展为数组
                                if data_length:
                                    arrays.append(np.full(data_length, elem))
                                else:
                                    arrays.append(np.array([elem]))

                        # 计算平均值作为部分表达式的估值
                        result_array = np.mean(arrays, axis=0)

                        if data_index is not None:
                            return pd.Series(result_array, index=data_index)
                        else:
                            return result_array
                    except Exception as e:
                        logger.debug(f"Failed to combine stack elements: {e}")
                        # 失败时返回栈顶元素
                        pass

                # 返回结果
                if isinstance(result, (int, float, np.number)):
                    if data_length and data_index is not None:
                        return pd.Series(result, index=data_index)
                    elif data_length:
                        return np.full(data_length, result)
                return result
            else:
                # 不允许部分表达式时报错
                logger.error(f"Stack has {len(stack)} elements after evaluation, expected 1")
                logger.error(f"Stack content: {[type(x) for x in stack]}")
                logger.error(f"RPN expression: {' '.join([t.name for t in token_sequence])}")
                return None

    @staticmethod
    def apply_unary_op(op_name, operand, data_length=None, data_index=None):
        """应用一元操作符 - 确保返回向量"""
        if isinstance(operand, (int, float)) and data_length:
            if data_index is not None:
                operand = pd.Series(operand, index=data_index)
            else:
                operand = np.full(data_length, operand)

        if op_name == 'abs':
            return np.abs(operand)
        elif op_name == 'log':
            # 安全的log操作
            if isinstance(operand, pd.Series):
                return np.log(np.maximum(operand.abs() + 1e-10, 1e-10))
            else:
                return np.log(np.maximum(np.abs(operand) + 1e-10, 1e-10))
        elif op_name == 'sign':
            # Sign: 返回1如果为正，否则返回0（按照论文）
            return np.where(operand > 0, 1.0, 0.0)
        elif op_name == 'csrank':
            # 横截面排名
            return RPNEvaluator.apply_cross_section_op('csrank', operand)

        else:
            raise ValueError(f"Unknown unary operator: {op_name}")

    @staticmethod
    def apply_binary_op(op_name, operand1, operand2, data_length=None, data_index=None):
        """应用二元操作符"""

        # 确保两个操作数都是相同长度的向量
        if isinstance(operand1, (int, float)) and isinstance(operand2, (pd.Series, np.ndarray)):
            if isinstance(operand2, pd.Series):
                operand1 = pd.Series(operand1, index=operand2.index)
            else:
                operand1 = np.full(len(operand2), operand1)
        elif isinstance(operand2, (int, float)) and isinstance(operand1, (pd.Series, np.ndarray)):
            if isinstance(operand1, pd.Series):
                operand2 = pd.Series(operand2, index=operand1.index)
            else:
                operand2 = np.full(len(operand1), operand2)

        if op_name == 'add':
            return operand1 + operand2
        elif op_name == 'sub':
            return operand1 - operand2
        elif op_name == 'mul':
            return operand1 * operand2
        elif op_name == 'div':
            # 安全除法
            if isinstance(operand1, pd.Series):
                return operand1.div(operand2).replace([np.inf, -np.inf], 0).fillna(0)
            else:
                return np.divide(operand1, operand2, out=np.zeros_like(operand1), where=operand2 != 0)
        elif op_name == 'greater':
            return (operand1 > operand2).astype(float)
        elif op_name == 'less':
            return (operand1 < operand2).astype(float)
        else:
            raise ValueError(f"Unknown binary operator: {op_name}")

    @staticmethod
    def apply_cross_section_op(op_name, data):
        """应用横截面操作符"""
        if op_name == 'csrank':
            # CSRank: 横截面排名（相对于当天所有股票）
            if isinstance(data, pd.Series):
                # 如果有多级索引(ticker, date)，按date分组排名
                if isinstance(data.index, pd.MultiIndex):
                    return data.groupby(level=1).rank(pct=True)
                else:
                    # 单级索引，直接排名
                    return data.rank(pct=True)
            else:
                # NumPy数组
                return stats.rankdata(data, method='average') / len(data)
        else:
            raise ValueError(f"Unknown cross-section operator: {op_name}")

    @staticmethod
    def apply_time_series_op(op_name, data, window):
        """应用所有时序操作符（包括论文中的完整列表）"""
        # 确保window是整数
        if isinstance(window, (pd.Series, np.ndarray)):
            window = int(window[0]) if len(window) > 0 else 5
        else:
            window = int(window)

        window = max(1, min(window, 100))  # 限制窗口大小

        if isinstance(data, pd.Series):
            if op_name == 'ts_ref':
                # Ref(x,t): t天前的值
                return data.shift(window)

            elif op_name == 'ts_rank':
                # Rank(x,t): 当前值在过去t天中的排名
                return data.rolling(window=window, min_periods=1).apply(
                    lambda x: stats.rankdata(x)[-1] / len(x)
                )

            elif op_name == 'ts_mean':
                # Mean(x,t): 过去t天的平均值
                return data.rolling(window=window, min_periods=1).mean()

            elif op_name == 'ts_med':
                # Med(x,t): 过去t天的中位数
                return data.rolling(window=window, min_periods=1).median()

            elif op_name == 'ts_sum':
                # Sum(x,t): 过去t天的总和
                return data.rolling(window=window, min_periods=1).sum()

            elif op_name == 'ts_std':
                # Std(x,t): 过去t天的标准差
                # 关键修复：动态调整min_periods
                min_periods_required = min(2, window)
                if window < 2:
                    # 窗口太小，返回0或NaN
                    return pd.Series(0, index=data.index)
                return data.rolling(window=window, min_periods=min_periods_required).std().fillna(0)

            elif op_name == 'ts_var':
                # Var(x,t): 过去t天的方差
                # 关键修复：动态调整min_periods
                min_periods_required = min(2, window)
                if window < 2:
                    # 窗口太小，返回0
                    return pd.Series(0, index=data.index)
                return data.rolling(window=window, min_periods=min_periods_required).var().fillna(0)

            elif op_name == 'ts_max':
                # Max(x,t): 过去t天的最大值
                return data.rolling(window=window, min_periods=1).max()

            elif op_name == 'ts_min':
                # Min(x,t): 过去t天的最小值
                return data.rolling(window=window, min_periods=1).min()

            elif op_name == 'ts_skew':
                # Skew(x,t): 过去t天的偏度
                # 关键修复：动态调整min_periods
                min_periods_required = min(3, window)
                if window < 3:
                    # 窗口太小，返回0
                    return pd.Series(0, index=data.index)
                return data.rolling(window=window, min_periods=min_periods_required).skew().fillna(0)

            elif op_name == 'ts_kurt':
                # Kurt(x,t): 过去t天的峰度
                # 关键修复：动态调整min_periods
                min_periods_required = min(4, window)
                if window < 4:
                    # 窗口太小，返回0
                    return pd.Series(0, index=data.index)
                return data.rolling(window=window, min_periods=min_periods_required).kurt().fillna(0)

            elif op_name == 'ts_wma':
                # WMA(x,t): 加权移动平均（线性权重）
                weights = np.arange(1, window + 1)
                weights = weights / weights.sum()

                def weighted_mean(x):
                    if len(x) < window:
                        w = weights[:len(x)]
                        w = w / w.sum()
                    else:
                        w = weights
                    return np.dot(x, w)

                return data.rolling(window=window, min_periods=1).apply(weighted_mean)

            elif op_name == 'ts_ema':
                # EMA(x,t): 指数移动平均
                return data.ewm(span=window, adjust=False, min_periods=1).mean()

            else:
                raise ValueError(f"Unknown time series operator: {op_name}")

        else:
            # NumPy数组处理（保持原有逻辑，但加入相同的修复）
            result = np.zeros_like(data)

            for i in range(len(data)):
                start_idx = max(0, i - window + 1)
                window_data = data[start_idx:i + 1]

                if op_name == 'ts_ref':
                    if i >= window:
                        result[i] = data[i - window]
                    else:
                        result[i] = np.nan

                elif op_name == 'ts_rank':
                    # 优化版本：避免使用lambda函数
                    def rank_last(x):
                        if len(x) == 0:
                            return 0.5
                        try:
                            # 使用更高效的方式计算排名
                            return (x.iloc[-1] > x).sum() / len(x)
                        except:
                            return 0.5

                    # 对于大数据集，考虑分块处理
                    if len(data) > 10000:
                        # 分块处理避免内存溢出
                        chunk_size = 5000
                        results = []
                        for i in range(0, len(data), chunk_size):
                            chunk = data.iloc[i:min(i + chunk_size, len(data))]
                            result = chunk.rolling(window=window, min_periods=1).apply(
                                rank_last, raw=False
                            )
                            results.append(result)
                        return pd.concat(results)
                    else:
                        return data.rolling(window=window, min_periods=1).apply(
                            rank_last, raw=False
                        )

                elif op_name == 'ts_mean':
                    result[i] = np.mean(window_data)

                elif op_name == 'ts_med':
                    result[i] = np.median(window_data)

                elif op_name == 'ts_sum':
                    result[i] = np.sum(window_data)

                elif op_name == 'ts_std':
                    # 修复：检查窗口大小
                    if len(window_data) > 1 and window >= 2:
                        result[i] = np.std(window_data)
                    else:
                        result[i] = 0

                elif op_name == 'ts_var':
                    # 修复：检查窗口大小
                    if len(window_data) > 1 and window >= 2:
                        result[i] = np.var(window_data)
                    else:
                        result[i] = 0

                elif op_name == 'ts_max':
                    result[i] = np.max(window_data)

                elif op_name == 'ts_min':
                    result[i] = np.min(window_data)

                elif op_name == 'ts_skew':
                    # 修复：检查窗口大小
                    if len(window_data) >= 3 and window >= 3:
                        result[i] = stats.skew(window_data)
                    else:
                        result[i] = 0

                elif op_name == 'ts_kurt':
                    # 修复：检查窗口大小
                    if len(window_data) >= 4 and window >= 4:
                        result[i] = stats.kurtosis(window_data)
                    else:
                        result[i] = 0

                elif op_name == 'ts_wma':
                    weights = np.arange(1, len(window_data) + 1)
                    weights = weights / weights.sum()
                    result[i] = np.dot(window_data, weights)

                elif op_name == 'ts_ema':
                    alpha = 2.0 / (window + 1)
                    if i == 0:
                        result[i] = data[i]
                    else:
                        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

            return result

    @staticmethod
    def apply_ternary_op(op_name, operand1, operand2, operand3):
        """应用三元操作符"""
        if op_name == 'corr':
            # 相关性：operand1和operand2的相关性，窗口大小为operand3
            window = int(operand3) if isinstance(operand3, (int, float)) else int(operand3[0])
            window = max(2, min(window, 100))

            if isinstance(operand1, pd.Series) and isinstance(operand2, pd.Series):
                return operand1.rolling(window=window, min_periods=2).corr(operand2)
            else:
                # NumPy实现
                result = np.zeros(len(operand1))
                for i in range(len(operand1)):
                    start_idx = max(0, i - window + 1)
                    if i - start_idx >= 1:  # 至少需要2个点
                        corr = np.corrcoef(operand1[start_idx:i + 1], operand2[start_idx:i + 1])[0, 1]
                        result[i] = corr if not np.isnan(corr) else 0
                return result

        elif op_name == 'cov':
            # 协方差
            window = int(operand3) if isinstance(operand3, (int, float)) else int(operand3[0])
            window = max(2, min(window, 100))

            if isinstance(operand1, pd.Series) and isinstance(operand2, pd.Series):
                return operand1.rolling(window=window, min_periods=2).cov(operand2)
            else:
                # NumPy实现
                result = np.zeros(len(operand1))
                for i in range(len(operand1)):
                    start_idx = max(0, i - window + 1)
                    if i - start_idx >= 1:
                        cov = np.cov(operand1[start_idx:i + 1], operand2[start_idx:i + 1])[0, 1]
                        result[i] = cov if not np.isnan(cov) else 0
                return result
        else:
            raise ValueError(f"Unknown ternary operator: {op_name}")

    @staticmethod
    def tokens_to_infix(token_sequence):
        """将RPN Token序列转换为中缀表达式字符串（用于可读性）"""
        stack = []

        for token in token_sequence[1:]:  # 跳过BEG
            if token.name == 'END':
                break

            if token.type == TokenType.OPERAND:
                # 操作数直接入栈
                stack.append(token.name)

            elif token.type == TokenType.OPERATOR:
                if token.arity == 1:
                    # 一元操作符
                    if len(stack) >= 1:
                        operand = stack.pop()
                        stack.append(f"{token.name}({operand})")

                elif token.arity == 2:
                    # 二元操作符
                    if len(stack) >= 2:
                        right = stack.pop()
                        left = stack.pop()

                        if token.name in ['add', 'sub', 'mul', 'div']:
                            # 算术操作符用中缀表示
                            op_symbol = {
                                'add': '+', 'sub': '-',
                                'mul': '*', 'div': '/'
                            }.get(token.name, token.name)
                            stack.append(f"({left} {op_symbol} {right})")
                        else:
                            # 其他用函数表示
                            stack.append(f"{token.name}({left}, {right})")

                elif token.arity == 3:
                    # 三元操作符
                    if len(stack) >= 3:
                        arg3 = stack.pop()
                        arg2 = stack.pop()
                        arg1 = stack.pop()
                        stack.append(f"{token.name}({arg1}, {arg2}, {arg3})")

        return stack[0] if stack else ""