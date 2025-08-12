"""RPN评估器测试文件"""
import pytest
import numpy as np
import pandas as pd
from core.rpn_evaluator import RPNEvaluator
from core.token_system import TOKEN_DEFINITIONS
from core.operators import Operators


class TestRPNEvaluator:
    """测试RPN评估器"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        n = 100
        data = {
            'open': pd.Series(np.random.randn(n) * 10 + 100),
            'high': pd.Series(np.random.randn(n) * 10 + 105),
            'low': pd.Series(np.random.randn(n) * 10 + 95),
            'close': pd.Series(np.random.randn(n) * 10 + 100),
            'volume': pd.Series(np.random.exponential(1000000, n)),
            'vwap': pd.Series(np.random.randn(n) * 10 + 100)
        }
        return data

    def test_simple_operand(self, sample_data):
        """测试简单操作数"""
        # BEG close END
        tokens = [
            TOKEN_DEFINITIONS['BEG'],
            TOKEN_DEFINITIONS['close'],
            TOKEN_DEFINITIONS['END']
        ]

        result = RPNEvaluator.evaluate(tokens, sample_data)

        assert result is not None
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data['close'])
        assert np.allclose(result.values, sample_data['close'].values)

    def test_binary_operation(self, sample_data):
        """测试二元运算"""
        # BEG close volume add END
        tokens = [
            TOKEN_DEFINITIONS['BEG'],
            TOKEN_DEFINITIONS['close'],
            TOKEN_DEFINITIONS['volume'],
            TOKEN_DEFINITIONS['add'],
            TOKEN_DEFINITIONS['END']
        ]

        result = RPNEvaluator.evaluate(tokens, sample_data)
        expected = sample_data['close'] + sample_data['volume']

        assert result is not None
        assert isinstance(result, pd.Series)
        assert np.allclose(result.values, expected.values)

    def test_unary_operation(self, sample_data):
        """测试一元运算"""
        # BEG close abs END
        tokens = [
            TOKEN_DEFINITIONS['BEG'],
            TOKEN_DEFINITIONS['close'],
            TOKEN_DEFINITIONS['abs'],
            TOKEN_DEFINITIONS['END']
        ]

        result = RPNEvaluator.evaluate(tokens, sample_data)
        expected = np.abs(sample_data['close'])

        assert result is not None
        assert np.allclose(result.values, expected.values)

    def test_time_series_operation(self, sample_data):
        """测试时序操作"""
        # BEG close ts_mean delta_5 END
        tokens = [
            TOKEN_DEFINITIONS['BEG'],
            TOKEN_DEFINITIONS['close'],
            TOKEN_DEFINITIONS['ts_mean'],
            TOKEN_DEFINITIONS['delta_5'],
            TOKEN_DEFINITIONS['END']
        ]

        result = RPNEvaluator.evaluate(tokens, sample_data)

        assert result is not None
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data['close'])

        # 验证移动平均的正确性
        expected = sample_data['close'].rolling(window=5, min_periods=1).mean()
        assert np.allclose(result.values, expected.values, rtol=1e-5)

    def test_constant_operation(self, sample_data):
        """测试常数操作"""
        # BEG close const_2 mul END
        tokens = [
            TOKEN_DEFINITIONS['BEG'],
            TOKEN_DEFINITIONS['close'],
            TOKEN_DEFINITIONS['const_2'],
            TOKEN_DEFINITIONS['mul'],
            TOKEN_DEFINITIONS['END']
        ]

        result = RPNEvaluator.evaluate(tokens, sample_data)
        expected = sample_data['close'] * 2

        assert result is not None
        assert np.allclose(result.values, expected.values)

    def test_division_by_zero(self, sample_data):
        """测试除零处理"""
        # 创建包含零的数据
        data_with_zero = sample_data.copy()
        data_with_zero['zero'] = pd.Series([0] * len(sample_data['close']))

        # BEG close zero div END
        tokens = [
            TOKEN_DEFINITIONS['BEG'],
            TOKEN_DEFINITIONS['close'],
            TOKEN_DEFINITIONS['zero'],
            TOKEN_DEFINITIONS['div'],
            TOKEN_DEFINITIONS['END']
        ]

        result = RPNEvaluator.evaluate(tokens, data_with_zero)

        assert result is not None
        # 除零应该返回0而不是inf
        assert not np.any(np.isinf(result.values))
        assert np.all(result.values == 0)

    def test_partial_expression(self, sample_data):
        """测试部分表达式"""
        # BEG close volume （没有END，栈中有2个元素）
        tokens = [
            TOKEN_DEFINITIONS['BEG'],
            TOKEN_DEFINITIONS['close'],
            TOKEN_DEFINITIONS['volume']
        ]

        # allow_partial=True时应该返回结果
        result = RPNEvaluator.evaluate(tokens, sample_data, allow_partial=True)
        assert result is not None

        # allow_partial=False时应该返回None
        result = RPNEvaluator.evaluate(tokens, sample_data, allow_partial=False)
        assert result is None

    def test_correlation_operation(self, sample_data):
        """测试相关性计算"""
        # BEG close volume corr delta_10 END
        tokens = [
            TOKEN_DEFINITIONS['BEG'],
            TOKEN_DEFINITIONS['close'],
            TOKEN_DEFINITIONS['volume'],
            TOKEN_DEFINITIONS['corr'],
            TOKEN_DEFINITIONS['delta_10'],
            TOKEN_DEFINITIONS['END']
        ]

        result = RPNEvaluator.evaluate(tokens, sample_data)

        assert result is not None
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data['close'])

        # 相关系数应该在[-1, 1]之间
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= -1)
        assert np.all(valid_values <= 1)

    def test_complex_expression(self, sample_data):
        """测试复杂表达式"""
        # BEG close volume add ts_mean delta_5 high low sub div END
        tokens = [
            TOKEN_DEFINITIONS['BEG'],
            TOKEN_DEFINITIONS['close'],
            TOKEN_DEFINITIONS['volume'],
            TOKEN_DEFINITIONS['add'],
            TOKEN_DEFINITIONS['ts_mean'],
            TOKEN_DEFINITIONS['delta_5'],
            TOKEN_DEFINITIONS['high'],
            TOKEN_DEFINITIONS['low'],
            TOKEN_DEFINITIONS['sub'],
            TOKEN_DEFINITIONS['div'],
            TOKEN_DEFINITIONS['END']
        ]

        result = RPNEvaluator.evaluate(tokens, sample_data)

        assert result is not None
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data['close'])

        # 结果不应该全是NaN
        assert not result.isna().all()

    def test_empty_data(self):
        """测试空数据"""
        empty_data = {}

        tokens = [
            TOKEN_DEFINITIONS['BEG'],
            TOKEN_DEFINITIONS['const_1'],
            TOKEN_DEFINITIONS['END']
        ]

        result = RPNEvaluator.evaluate(tokens, empty_data)
        assert result is not None
        assert result == 1.0

    def test_nan_handling(self, sample_data):
        """测试NaN处理"""
        # 添加NaN值
        data_with_nan = sample_data.copy()
        data_with_nan['close'].iloc[0:10] = np.nan

        tokens = [
            TOKEN_DEFINITIONS['BEG'],
            TOKEN_DEFINITIONS['close'],
            TOKEN_DEFINITIONS['ts_mean'],
            TOKEN_DEFINITIONS['delta_5'],
            TOKEN_DEFINITIONS['END']
        ]

        result = RPNEvaluator.evaluate(tokens, data_with_nan)

        assert result is not None
        # 应该正确处理NaN，不应该传播到所有值
        assert not result.isna().all()
        assert result.iloc[15:].notna().any()  # 后面的值应该有非NaN

    def test_data_alignment(self):
        """测试数据对齐"""
        # 创建不同索引的数据
        data = {
            'close': pd.Series([100, 101, 102], index=[0, 1, 2]),
            'volume': pd.Series([1000, 2000, 3000], index=[1, 2, 3])
        }

        tokens = [
            TOKEN_DEFINITIONS['BEG'],
            TOKEN_DEFINITIONS['close'],
            TOKEN_DEFINITIONS['volume'],
            TOKEN_DEFINITIONS['add'],
            TOKEN_DEFINITIONS['END']
        ]

        result = RPNEvaluator.evaluate(tokens, data)

        assert result is not None
        # 结果索引应该是交集
        assert len(result) == 3  # 保持原始长度

    def test_invalid_token_sequence(self):
        """测试无效的Token序列"""
        sample_data = {'close': pd.Series([100, 101, 102])}

        # 缺少操作数的序列
        tokens = [
            TOKEN_DEFINITIONS['BEG'],
            TOKEN_DEFINITIONS['add'],  # 需要2个操作数
            TOKEN_DEFINITIONS['END']
        ]

        result = RPNEvaluator.evaluate(tokens, sample_data, allow_partial=False)
        assert result is None

    def test_dynamic_constant(self):
        """测试动态常数"""
        sample_data = {'close': pd.Series([100, 101, 102])}

        # 创建动态常数Token
        from core.token_system import Token, TokenType
        dynamic_const = Token(TokenType.OPERAND, 'const_3.14', value=3.14)

        tokens = [
            TOKEN_DEFINITIONS['BEG'],
            TOKEN_DEFINITIONS['close'],
            dynamic_const,
            TOKEN_DEFINITIONS['mul'],
            TOKEN_DEFINITIONS['END']
        ]

        result = RPNEvaluator.evaluate(tokens, sample_data)
        expected = sample_data['close'] * 3.14

        assert result is not None
        assert np.allclose(result.values, expected.values)


class TestOperators:
    """测试操作符实现"""

    def test_safe_divide(self):
        """测试安全除法"""
        # Series除法
        s1 = pd.Series([10, 20, 30])
        s2 = pd.Series([2, 0, 5])
        result = Operators.safe_divide(s1, s2, default_value=-1)

        assert result.iloc[0] == 5.0
        assert result.iloc[1] == -1  # 除零返回默认值
        assert result.iloc[2] == 6.0

        # NumPy数组除法
        a1 = np.array([10, 20, 30])
        a2 = np.array([2, 0, 5])
        result = Operators.safe_divide(a1, a2, default_value=-1)

        assert result[0] == 5.0
        assert result[1] == -1
        assert result[2] == 6.0

    def test_csrank(self):
        """测试横截面排名"""
        data = pd.Series([10, 30, 20, 40])
        result = Operators.csrank(data)

        # 排名百分位应该在[0, 1]之间
        assert np.all(result >= 0)
        assert np.all(result <= 1)

        # 最大值应该排名最高
        assert result.iloc[3] == result.max()
        # 最小值应该排名最低
        assert result.iloc[0] == result.min()

    def test_ts_std_small_window(self):
        """测试小窗口标准差"""
        data = pd.Series([1, 2, 3, 4, 5])

        # 窗口小于3时的特殊处理
        result = Operators.ts_std(data, 2)
        assert result is not None
        assert not result.isna().all()

        # 窗口为1时
        result = Operators.ts_std(data, 1)
        assert result is not None
        # 窗口为1时标准差应该是0或很小
        assert result.iloc[0] == 0

    def test_ts_skew_kurt_small_window(self):
        """测试小窗口偏度和峰度"""
        data = pd.Series(np.random.randn(20))

        # 窗口小于5时返回0
        result = Operators.ts_skew(data, 3)
        assert np.all(result == 0)

        result = Operators.ts_kurt(data, 4)
        assert np.all(result == 0)

        # 窗口足够大时应该有非零值
        result = Operators.ts_skew(data, 10)
        assert not np.all(result == 0)

    def test_correlation_minimum_points(self):
        """测试相关性计算最少点数"""
        data1 = pd.Series([1, 2])
        data2 = pd.Series([2, 4])

        # 少于2个点时返回0
        result = Operators.corr(data1[:1], data2[:1], 5)
        assert result[0] == 0

        # 正好2个点时可以计算
        result = Operators.corr(data1, data2, 2)
        assert result[1] != 0  # 第二个点应该有相关性

    def test_align_operands(self):
        """测试操作数对齐"""
        # 标量与Series
        s = pd.Series([1, 2, 3])
        op1, op2 = Operators._align_operands(5, s)

        assert isinstance(op1, pd.Series)
        assert len(op1) == len(s)
        assert np.all(op1 == 5)

        # 标量与数组
        a = np.array([1, 2, 3])
        op1, op2 = Operators._align_operands(5, a)

        assert isinstance(op1, np.ndarray)
        assert len(op1) == len(a)
        assert np.all(op1 == 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])