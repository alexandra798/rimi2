"""RPN表达式求值器"""
import numpy as np
import pandas as pd
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
        评估RPN表达式 支持部分表达式
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
                    # 始终创建 Series
                    if data_index is not None:
                        stack.append(pd.Series(const_value, index=data_index))
                    elif data_length:
                        # 即使没有索引，也创建 Series
                        stack.append(pd.Series([const_value] * data_length))
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
                    else:
                        i += 1
                    result = RPNEvaluator.apply_time_series_op(
                        token.name, data_operand, window
                    )
                    stack.append(result)
                    continue  # 跳过主循环的 i += 1



                elif token.name in ('corr', 'cov'):

                    if len(stack) < 2:
                        logger.error(f"Insufficient operands for {token.name}")
                        return None

                    y = stack.pop()
                    x = stack.pop()

                    window = 5  # 默认窗口
                    if i + 1 < len(token_sequence) and token_sequence[i + 1].name.startswith('delta_'):
                        delta_token = token_sequence[i + 1]
                        window = int(delta_token.name.split('_')[1])
                        i += 2  # 跳过操作符后的 delta_*
                    else:
                        i += 1  # 没有 delta_*，使用默认窗口，并前进一个 token

                    # 直接把窗口当作第三参传给三元实现（第三参现在是 int）
                    result = RPNEvaluator.apply_ternary_op(token.name, x, y, window)
                    stack.append(result)
                    continue  # 已手动推进 i

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
        if isinstance(operand, (int, float)) and data_length:
            # 始终创建 Series 而不是 numpy 数组
            if data_index is not None:
                operand = pd.Series(operand, index=data_index)
            else:
                operand = pd.Series([operand] * data_length)


        if op_name == 'abs':
            return np.abs(operand)
        elif op_name == 'log':
            # 安全的log操作
            if isinstance(operand, pd.Series):
                return np.log(np.maximum(operand.abs() + 1e-10, 1e-10))
            else:
                return np.log(np.maximum(np.abs(operand) + 1e-10, 1e-10))
        elif op_name == 'sign':
            # 如果是 Series，保留索引；按论文定义：正数=1，非正=0
            if isinstance(operand, pd.Series):
                return (operand > 0).astype(float)
            else:
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
        """应用所有时序操作符"""
        # 确保window是整数
        if isinstance(window, (pd.Series, np.ndarray)):
            window = int(window[0]) if len(window) > 0 else 5
        else:
            window = int(window)

        window = max(1, min(window, 100))  # 限制窗口大小

        # ================== Pandas Series 处理 ==================
        if isinstance(data, pd.Series):
            return RPNEvaluator._apply_time_series_pandas(op_name, data, window)

        # ================== NumPy Array 处理 ==================
        else:
            return RPNEvaluator._apply_time_series_numpy(op_name, data, window)

    @staticmethod
    def _apply_time_series_pandas(op_name, data, window):
        """Pandas Series的时序操作实现"""

        if op_name == 'ts_ref':
            return data.shift(window)

        elif op_name == 'ts_rank':
            def rank_in_window(x):
                if len(x) < 2:
                    return 0.5
                return (x.iloc[-1] > x).sum() / len(x)

            return data.rolling(window=window, min_periods=1).apply(rank_in_window, raw=False)

        elif op_name == 'ts_mean':
            return data.rolling(window=window, min_periods=1).mean()

        elif op_name == 'ts_med':
            return data.rolling(window=window, min_periods=1).median()

        elif op_name == 'ts_sum':
            return data.rolling(window=window, min_periods=1).sum()

        elif op_name == 'ts_std' or op_name == 'ts_var':
            if window < 3:
                # 不返回常数，使用扩展差分作为波动性度量
                diff = data.diff().abs()
                result = diff.rolling(window=max(window, 2), min_periods=1).mean()
                if op_name == 'ts_var':
                    result = result ** 2
                return result
            else:
                min_periods = min(3, window)
                result = data.rolling(window=window, min_periods=min_periods).std()
                if op_name == 'ts_var':
                    result = result ** 2
                return result.bfill().fillna(0)

        elif op_name == 'ts_max':
            return data.rolling(window=window, min_periods=1).max()

        elif op_name == 'ts_min':
            return data.rolling(window=window, min_periods=1).min()

        elif op_name == 'ts_skew':
            if window < 5:
                # 简化的偏度估计
                mean = data.rolling(window=max(window, 3), min_periods=1).mean()
                deviation = data - mean
                pos_dev = deviation.where(deviation > 0, 0)
                neg_dev = deviation.where(deviation < 0, 0).abs()
                skew_proxy = (pos_dev.rolling(window=max(window, 3), min_periods=1).sum() -
                              neg_dev.rolling(window=max(window, 3), min_periods=1).sum())
                return skew_proxy / (data.rolling(window=max(window, 3), min_periods=1).std() + 1e-8)
            else:
                min_periods = min(5, window)
                return data.rolling(window=window, min_periods=min_periods).skew().fillna(0)

        elif op_name == 'ts_kurt':
            if window < 5:
                # 简化的峰度估计
                std = data.rolling(window=max(window, 3), min_periods=1).std()
                mean = data.rolling(window=max(window, 3), min_periods=1).mean()
                normalized = (data - mean) / (std + 1e-8)
                kurt_proxy = (normalized.abs() > 2).rolling(window=max(window, 3), min_periods=1).mean()
                return kurt_proxy * 10
            else:
                min_periods = min(5, window)
                return data.rolling(window=window, min_periods=min_periods).kurt().fillna(0)

        elif op_name == 'ts_wma':
            # 加权移动平均，忽略 NaN 并重标化权重
            weights = np.arange(1, window + 1, dtype=np.float64)

            def weighted_mean(x):
                # x 是 ndarray（raw=True）
                x = np.asarray(x, dtype=float)
                m = ~np.isnan(x)
                if not m.any():
                    return 0.0
                x = x[m]
                w = weights[:len(x)]
                w = w / w.sum()
                return float(np.dot(x, w))

            return data.rolling(window=window, min_periods=1).apply(weighted_mean, raw=True)


        elif op_name == 'ts_ema':
            return data.ewm(span=window, adjust=False, min_periods=1).mean()

        else:
            raise ValueError(f"Unknown time series operator: {op_name}")

    @staticmethod
    def _apply_time_series_numpy(op_name, data, window):
        """
        NumPy Array的时序操作实现 - 完整版本
        避免产生常数，使用智能替代方法
        """

        # 确保是numpy数组
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # 初始化结果数组
        result = np.zeros_like(data, dtype=np.float64)
        data_len = len(data)

        # ================== ts_ref: 引用t天前的值 ==================
        if op_name == 'ts_ref':
            result[:window] = np.nan  # 前window个值无法引用
            if window < data_len:
                result[window:] = data[:-window]
            return result

        # ================== ts_rank: 窗口内排名 ==================
        elif op_name == 'ts_rank':
            for i in range(data_len):
                start_idx = max(0, i - window + 1)
                window_data = data[start_idx:i + 1]

                if len(window_data) < 2:
                    result[i] = 0.5
                else:
                    # 计算当前值在窗口中的排名百分位
                    current_val = data[i]
                    rank = (current_val > window_data).sum() / len(window_data)
                    result[i] = rank
            return result

        # ================== ts_mean: 移动平均 ==================
        elif op_name == 'ts_mean':
            for i in range(data_len):
                start_idx = max(0, i - window + 1)
                window_data = data[start_idx:i + 1]
                result[i] = np.mean(window_data)
            return result

        # ================== ts_med: 移动中位数 ==================
        elif op_name == 'ts_med':
            for i in range(data_len):
                start_idx = max(0, i - window + 1)
                window_data = data[start_idx:i + 1]
                result[i] = np.median(window_data)
            return result

        # ================== ts_sum: 移动求和 ==================
        elif op_name == 'ts_sum':
            for i in range(data_len):
                start_idx = max(0, i - window + 1)
                window_data = data[start_idx:i + 1]
                result[i] = np.sum(window_data)
            return result

        # ================== ts_std: 标准差（智能处理小窗口）==================
        elif op_name == 'ts_std':
            if window < 3:
                # 小窗口：使用移动差分的绝对值作为波动性度量
                diff = np.diff(data, prepend=data[0])
                abs_diff = np.abs(diff)

                for i in range(data_len):
                    start_idx = max(0, i - max(window, 2) + 1)
                    window_diff = abs_diff[start_idx:i + 1]
                    result[i] = np.mean(window_diff) if len(window_diff) > 0 else 0
            else:
                # 正常窗口：计算标准差
                for i in range(data_len):
                    start_idx = max(0, i - window + 1)
                    window_data = data[start_idx:i + 1]

                    if len(window_data) >= 2:
                        result[i] = np.std(window_data, ddof=1)  # 使用样本标准差
                    else:
                        # 单个数据点，使用与前一个值的差作为估计
                        if i > 0:
                            result[i] = abs(data[i] - data[i - 1]) / np.sqrt(2)
                        else:
                            result[i] = 0
            return result

        # ================== ts_var: 方差（智能处理小窗口）==================
        elif op_name == 'ts_var':
            if window < 3:
                # 小窗口：使用差分方差估计
                diff = np.diff(data, prepend=data[0])

                for i in range(data_len):
                    start_idx = max(0, i - max(window, 2) + 1)
                    window_diff = diff[start_idx:i + 1]
                    if len(window_diff) > 0:
                        result[i] = np.var(window_diff)
                    else:
                        result[i] = 0
            else:
                # 正常窗口：计算方差
                for i in range(data_len):
                    start_idx = max(0, i - window + 1)
                    window_data = data[start_idx:i + 1]

                    if len(window_data) >= 2:
                        result[i] = np.var(window_data, ddof=1)  # 使用样本方差
                    else:
                        if i > 0:
                            result[i] = ((data[i] - data[i - 1]) / np.sqrt(2)) ** 2
                        else:
                            result[i] = 0
            return result

        # ================== ts_max: 移动最大值 ==================
        elif op_name == 'ts_max':
            for i in range(data_len):
                start_idx = max(0, i - window + 1)
                window_data = data[start_idx:i + 1]
                result[i] = np.max(window_data)
            return result

        # ================== ts_min: 移动最小值 ==================
        elif op_name == 'ts_min':
            for i in range(data_len):
                start_idx = max(0, i - window + 1)
                window_data = data[start_idx:i + 1]
                result[i] = np.min(window_data)
            return result

        # ================== ts_skew: 偏度（智能处理小窗口）==================
        elif op_name == 'ts_skew':
            for i in range(data_len):
                if window < 5:
                    # 小窗口：使用简化的偏度估计
                    start_idx = max(0, i - max(window, 3) + 1)
                    window_data = data[start_idx:i + 1]

                    if len(window_data) >= 3:
                        mean = np.mean(window_data)
                        std = np.std(window_data)

                        if std > 1e-8:
                            deviation = window_data - mean
                            # 计算正负偏离的不对称性
                            pos_dev = np.sum(deviation[deviation > 0])
                            neg_dev = np.sum(np.abs(deviation[deviation < 0]))

                            if pos_dev + neg_dev > 0:
                                skew_proxy = (pos_dev - neg_dev) / (pos_dev + neg_dev)
                                result[i] = skew_proxy * 3  # 缩放到合理范围
                            else:
                                result[i] = 0
                        else:
                            result[i] = 0
                    else:
                        result[i] = 0
                else:
                    # 正常窗口：计算标准偏度
                    start_idx = max(0, i - window + 1)
                    window_data = data[start_idx:i + 1]

                    if len(window_data) >= 3:
                        try:
                            if np.std(window_data) < 1e-10:
                                result[i] = 0
                            else:
                                val = stats.skew(window_data)
                                result[i] = 0 if (not np.isfinite(val)) else val
                        except Exception:
                            result[i] = 0
                    else:
                        result[i] = 0

            return result

        # ================== ts_kurt: 峰度（智能处理小窗口）==================
        elif op_name == 'ts_kurt':
            for i in range(data_len):
                if window < 5:
                    # 小窗口：使用简化的峰度估计
                    start_idx = max(0, i - max(window, 3) + 1)
                    window_data = data[start_idx:i + 1]

                    if len(window_data) >= 3:
                        mean = np.mean(window_data)
                        std = np.std(window_data)

                        if std > 1e-8:
                            normalized = (window_data - mean) / std
                            # 计算极端值的比例作为峰度代理
                            extreme_ratio = np.sum(np.abs(normalized) > 2) / len(normalized)
                            result[i] = extreme_ratio * 10  # 缩放
                        else:
                            result[i] = 0
                    else:
                        result[i] = 0
                else:
                    # 正常窗口：计算标准峰度
                    start_idx = max(0, i - window + 1)
                    window_data = data[start_idx:i + 1]

                    if len(window_data) >= 4:
                        try:
                            if np.std(window_data) < 1e-10:
                                result[i] = 0
                            else:
                                val = stats.kurtosis(window_data, fisher=True)
                                result[i] = 0 if (not np.isfinite(val)) else val
                        except Exception:
                            result[i] = 0
                    else:
                        result[i] = 0

            return result

        # ================== ts_wma: 加权移动平均 ==================
        elif op_name == 'ts_wma':
            # 线性权重；窗口内按有效样本重标化
            full_weights = np.arange(1, window + 1, dtype=np.float64)
            full_weights = full_weights / full_weights.sum()

            for i in range(data_len):
                start_idx = max(0, i - window + 1)
                window_data = data[start_idx:i + 1]
                if window_data.size == 0:
                    result[i] = 0.0
                    continue

                m = ~np.isnan(window_data)
                if not m.any():
                    # 没有有效样本：继承上一有效值或置0
                    result[i] = result[i - 1] if i > 0 and np.isfinite(result[i - 1]) else 0.0
                    continue

                valid = window_data[m]
                w = np.arange(1, len(valid) + 1, dtype=np.float64)
                w = w / w.sum()
                result[i] = float(np.dot(valid, w))
            return result


        # ================== ts_ema: 指数移动平均 ==================
        elif op_name == 'ts_ema':
            alpha = 2.0 / (window + 1)

            # 初始化第一个值
            result[0] = data[0]

            # 递归计算EMA
            for i in range(1, data_len):
                result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

            return result

        else:
            raise ValueError(f"Unknown time series operator: {op_name}")



    @staticmethod
    def apply_ternary_op(op_name, operand1, operand2, operand3):
        """应用三元操作符"""
        if op_name == 'corr':
            # 相关性：operand1和operand2的相关性，窗口大小为operand3
            window = int(operand3)
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
            window = int(operand3)
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