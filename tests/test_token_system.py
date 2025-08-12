"""Token系统测试文件"""
import pytest
import numpy as np
from core.token_system import (
    TokenType, Token, TOKEN_DEFINITIONS,
    TOKEN_TO_INDEX, RPNValidator
)


class TestTokenDefinitions:
    """测试Token定义"""

    def test_special_tokens(self):
        """测试特殊标记"""
        assert 'BEG' in TOKEN_DEFINITIONS
        assert 'END' in TOKEN_DEFINITIONS
        assert TOKEN_DEFINITIONS['BEG'].type == TokenType.SPECIAL
        assert TOKEN_DEFINITIONS['END'].type == TokenType.SPECIAL

    def test_operand_tokens(self):
        """测试操作数Token"""
        operands = ['open', 'high', 'low', 'close', 'volume', 'vwap']
        for op in operands:
            assert op in TOKEN_DEFINITIONS
            assert TOKEN_DEFINITIONS[op].type == TokenType.OPERAND

    def test_window_tokens(self):
        """测试时间窗口Token"""
        windows = ['delta_3', 'delta_5', 'delta_10', 'delta_20']
        for w in windows:
            assert w in TOKEN_DEFINITIONS
            assert TOKEN_DEFINITIONS[w].type == TokenType.OPERAND
            assert TOKEN_DEFINITIONS[w].value == int(w.split('_')[1])

    def test_operator_arity(self):
        """测试操作符元数"""
        # 一元操作符
        unary_ops = ['sign', 'abs', 'log', 'csrank']
        for op in unary_ops:
            assert TOKEN_DEFINITIONS[op].arity == 1

        # 二元操作符
        binary_ops = ['add', 'sub', 'mul', 'div', 'greater', 'less']
        for op in binary_ops:
            assert TOKEN_DEFINITIONS[op].arity == 2

    def test_time_series_operators(self):
        """测试时序操作符"""
        ts_ops = ['ts_mean', 'ts_std', 'ts_max', 'ts_min']
        for op in ts_ops:
            assert op in TOKEN_DEFINITIONS
            assert TOKEN_DEFINITIONS[op].type == TokenType.OPERATOR

            # 检查最小窗口要求
            if op in ['ts_std', 'ts_var']:
                assert TOKEN_DEFINITIONS[op].min_window == 3
            elif op in ['ts_skew', 'ts_kurt']:
                assert TOKEN_DEFINITIONS[op].min_window == 5


class TestRPNValidator:
    """测试RPN验证器"""

    def test_valid_partial_expression(self):
        """测试部分表达式验证"""
        # 有效的部分表达式
        valid_sequences = [
            [TOKEN_DEFINITIONS['BEG'], TOKEN_DEFINITIONS['close']],
            [TOKEN_DEFINITIONS['BEG'], TOKEN_DEFINITIONS['close'],
             TOKEN_DEFINITIONS['volume'], TOKEN_DEFINITIONS['add']],
        ]

        for seq in valid_sequences:
            assert RPNValidator.is_valid_partial_expression(seq) == True

    def test_invalid_partial_expression(self):
        """测试无效的部分表达式"""
        # 不以BEG开始
        invalid_seq = [TOKEN_DEFINITIONS['close']]
        assert RPNValidator.is_valid_partial_expression(invalid_seq) == False

        # 栈不平衡
        invalid_seq = [
            TOKEN_DEFINITIONS['BEG'],
            TOKEN_DEFINITIONS['add']  # 需要2个操作数
        ]
        assert RPNValidator.is_valid_partial_expression(invalid_seq) == False

    def test_calculate_stack_size(self):
        """测试栈大小计算"""
        # 简单序列
        seq = [
            TOKEN_DEFINITIONS['BEG'],
            TOKEN_DEFINITIONS['close'],
            TOKEN_DEFINITIONS['volume']
        ]
        assert RPNValidator.calculate_stack_size(seq) == 2

        # 带操作符的序列
        seq = [
            TOKEN_DEFINITIONS['BEG'],
            TOKEN_DEFINITIONS['close'],
            TOKEN_DEFINITIONS['volume'],
            TOKEN_DEFINITIONS['add']
        ]
        assert RPNValidator.calculate_stack_size(seq) == 1

    def test_get_valid_next_tokens(self):
        """测试获取合法的下一个Token"""
        # 初始状态
        seq = []
        valid = RPNValidator.get_valid_next_tokens(seq)
        assert valid == ['BEG']

        # BEG之后
        seq = [TOKEN_DEFINITIONS['BEG']]
        valid = RPNValidator.get_valid_next_tokens(seq)
        assert 'close' in valid
        assert 'volume' in valid
        assert 'END' not in valid  # 不能直接结束

        # 可以结束的状态
        seq = [
            TOKEN_DEFINITIONS['BEG'],
            TOKEN_DEFINITIONS['close']
        ]
        valid = RPNValidator.get_valid_next_tokens(seq)
        assert 'END' in valid

    def test_window_constraints(self):
        """测试窗口大小约束"""
        # ts_std需要至少3个点的窗口
        seq = [
            TOKEN_DEFINITIONS['BEG'],
            TOKEN_DEFINITIONS['close'],
            TOKEN_DEFINITIONS['ts_std']
        ]
        valid = RPNValidator.get_valid_next_tokens(seq)

        # delta_3应该满足最小窗口要求
        assert 'delta_3' in valid
        assert 'delta_5' in valid

        # ts_skew需要至少5个点
        seq = [
            TOKEN_DEFINITIONS['BEG'],
            TOKEN_DEFINITIONS['close'],
            TOKEN_DEFINITIONS['ts_skew']
        ]
        valid = RPNValidator.get_valid_next_tokens(seq)

        # delta_3不应该在有效列表中（小于5）
        assert 'delta_3' not in valid
        assert 'delta_5' in valid

    def test_can_terminate(self):
        """测试是否可以终止"""
        # 不能终止（栈为空）
        seq = [TOKEN_DEFINITIONS['BEG']]
        assert RPNValidator.can_terminate(seq) == False

        # 可以终止（栈中有1个元素）
        seq = [
            TOKEN_DEFINITIONS['BEG'],
            TOKEN_DEFINITIONS['close']
        ]
        assert RPNValidator.can_terminate(seq) == True

        # 不能终止（栈中有2个元素）
        seq = [
            TOKEN_DEFINITIONS['BEG'],
            TOKEN_DEFINITIONS['close'],
            TOKEN_DEFINITIONS['volume']
        ]
        assert RPNValidator.can_terminate(seq) == False

        # 可以终止（运算后栈中有1个元素）
        seq = [
            TOKEN_DEFINITIONS['BEG'],
            TOKEN_DEFINITIONS['close'],
            TOKEN_DEFINITIONS['volume'],
            TOKEN_DEFINITIONS['add']
        ]
        assert RPNValidator.can_terminate(seq) == True


class TestTokenEdgeCases:
    """测试Token系统的边界情况"""

    def test_max_sequence_length(self):
        """测试最大序列长度限制"""
        seq = [TOKEN_DEFINITIONS['BEG']]

        # 构建一个接近最大长度的序列
        for i in range(28):  # 加上BEG共29个
            seq.append(TOKEN_DEFINITIONS['close'])

        valid = RPNValidator.get_valid_next_tokens(seq)
        assert len(valid) > 0  # 还可以添加Token

        # 达到最大长度（30）
        seq.append(TOKEN_DEFINITIONS['close'])
        valid = RPNValidator.get_valid_next_tokens(seq)

        # 只能END（如果栈平衡）或空
        assert len(valid) == 0 or valid == ['END']

    def test_complex_expression(self):
        """测试复杂表达式"""
        # 构建：BEG close volume add const_2 mul ts_mean delta_5 END
        seq = [
            TOKEN_DEFINITIONS['BEG'],
            TOKEN_DEFINITIONS['close'],
            TOKEN_DEFINITIONS['volume'],
            TOKEN_DEFINITIONS['add'],
            TOKEN_DEFINITIONS['const_2'],
            TOKEN_DEFINITIONS['mul'],
            TOKEN_DEFINITIONS['ts_mean'],
            TOKEN_DEFINITIONS['delta_5'],
            TOKEN_DEFINITIONS['END']
        ]

        # 验证整个序列
        assert RPNValidator.is_valid_partial_expression(seq[:-1]) == True
        assert RPNValidator.can_terminate(seq[:-1]) == True

        # 最终栈大小应该是1
        assert RPNValidator.calculate_stack_size(seq[:-1]) == 1

    def test_correlation_operators(self):
        """测试相关性操作符"""
        # corr需要2个数据操作数和1个窗口参数
        seq = [
            TOKEN_DEFINITIONS['BEG'],
            TOKEN_DEFINITIONS['close'],
            TOKEN_DEFINITIONS['volume'],
            TOKEN_DEFINITIONS['corr']
        ]

        valid = RPNValidator.get_valid_next_tokens(seq)

        # 应该只返回满足最小窗口要求的delta
        for delta in valid:
            if delta.startswith('delta_'):
                value = int(delta.split('_')[1])
                assert value >= 3  # corr的最小窗口是3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])