"""测试所有算子的实现"""
import numpy as np
import pandas as pd
from core.operators import Operators
from core.rpn_evaluator import RPNEvaluator
from core.token_system import Token, TokenType, TOKEN_DEFINITIONS


def test_unary_operators():
    """测试一元操作符"""
    print("=" * 50)
    print("测试一元操作符")
    print("=" * 50)

    # 创建测试数据
    data = pd.Series([-5, -2, 0, 3, 7], index=['a', 'b', 'c', 'd', 'e'])
    data_np = np.array([-5, -2, 0, 3, 7])

    # 测试 abs
    print("\n1. 测试 abs:")
    result_pd = Operators.abs(data)
    result_np = Operators.abs(data_np)
    print(f"   Pandas结果: {result_pd.values}")
    print(f"   NumPy结果: {result_np}")

    # 测试 log
    print("\n2. 测试 log:")
    result_pd = Operators.log(data)
    result_np = Operators.log(data_np)
    print(f"   Pandas结果: {result_pd.values}")
    print(f"   NumPy结果: {result_np}")

    # 测试 sign
    print("\n3. 测试 sign:")
    result_pd = Operators.sign(data)
    result_np = Operators.sign(data_np)
    print(f"   Pandas结果: {result_pd.values}")
    print(f"   NumPy结果: {result_np}")

    # 测试 csrank
    print("\n4. 测试 csrank:")
    result_pd = Operators.csrank(data)
    result_np = Operators.csrank(data_np)
    print(f"   Pandas结果: {result_pd.values}")
    print(f"   NumPy结果: {result_np}")


def test_binary_operators():
    """测试二元操作符"""
    print("\n" + "=" * 50)
    print("测试二元操作符")
    print("=" * 50)

    # 创建测试数据
    data1 = pd.Series([1, 2, 3, 4, 5])
    data2 = pd.Series([2, 0, 3, 2, 1])

    # 测试加减乘除
    print("\n1. 测试 add:")
    result = Operators.add(data1, data2)
    print(f"   结果: {result.values}")

    print("\n2. 测试 sub:")
    result = Operators.sub(data1, data2)
    print(f"   结果: {result.values}")

    print("\n3. 测试 mul:")
    result = Operators.mul(data1, data2)
    print(f"   结果: {result.values}")

    print("\n4. 测试 div (包含除零):")
    result = Operators.div(data1, data2)
    print(f"   结果: {result.values}")

    print("\n5. 测试 greater:")
    result = Operators.greater(data1, data2)
    print(f"   结果: {result.values}")

    print("\n6. 测试 less:")
    result = Operators.less(data1, data2)
    print(f"   结果: {result.values}")


def test_time_series_operators():
    """测试时序操作符"""
    print("\n" + "=" * 50)
    print("测试时序操作符")
    print("=" * 50)

    # 创建测试数据 - 模拟股价数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=20)
    price = pd.Series(100 + np.cumsum(np.random.randn(20) * 2), index=dates)
    price_np = price.values

    window = 5

    # 测试各种时序操作符
    operators = [
        'ts_ref', 'ts_rank', 'ts_mean', 'ts_med', 'ts_sum',
        'ts_std', 'ts_var', 'ts_max', 'ts_min', 'ts_skew',
        'ts_kurt', 'ts_wma', 'ts_ema'
    ]

    for op_name in operators:
        print(f"\n测试 {op_name} (window={window}):")
        op_method = getattr(Operators, op_name)

        # Pandas版本
        result_pd = op_method(price, window)
        print(f"   Pandas结果前5个: {result_pd.iloc[:5].values}")

        # NumPy版本
        result_np = op_method(price_np, window)
        print(f"   NumPy结果前5个: {result_np[:5]}")


def test_correlation_operators():
    """测试相关性操作符"""
    print("\n" + "=" * 50)
    print("测试相关性操作符")
    print("=" * 50)

    # 创建相关的测试数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=20)
    x = pd.Series(np.random.randn(20), index=dates)
    y = x * 0.8 + np.random.randn(20) * 0.2  # 创建相关序列

    window = 5

    print(f"\n1. 测试 corr (window={window}):")
    result = Operators.corr(x, y, window)
    print(f"   结果前10个: {result.iloc[:10].values}")

    print(f"\n2. 测试 cov (window={window}):")
    result = Operators.cov(x, y, window)
    print(f"   结果前10个: {result.iloc[:10].values}")


def test_rpn_evaluation():
    """测试RPN表达式求值"""
    print("\n" + "=" * 50)
    print("测试RPN表达式求值")
    print("=" * 50)

    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=10)
    data_dict = {
        'open': pd.Series([100, 101, 99, 102, 103, 101, 104, 105, 103, 106], index=dates),
        'close': pd.Series([101, 100, 102, 103, 102, 104, 105, 104, 106, 107], index=dates),
        'volume': pd.Series([1000, 1100, 900, 1200, 1150, 1050, 1300, 1250, 1100, 1400], index=dates)
    }

    # 测试表达式 1: close - open (收益)
    print("\n1. 测试表达式: close open sub")
    tokens = [
        TOKEN_DEFINITIONS['BEG'],
        TOKEN_DEFINITIONS['close'],
        TOKEN_DEFINITIONS['open'],
        TOKEN_DEFINITIONS['sub'],
        TOKEN_DEFINITIONS['END']
    ]
    result = RPNEvaluator.evaluate(tokens, data_dict)
    print(f"   结果: {result.values}")

    # 测试表达式 2: ts_mean(close, 3)
    print("\n2. 测试表达式: close ts_mean delta_3")
    tokens = [
        TOKEN_DEFINITIONS['BEG'],
        TOKEN_DEFINITIONS['close'],
        TOKEN_DEFINITIONS['ts_mean'],
        TOKEN_DEFINITIONS['delta_3'],
        TOKEN_DEFINITIONS['END']
    ]
    result = RPNEvaluator.evaluate(tokens, data_dict)
    print(f"   结果: {result.values}")

    # 测试表达式 3: corr(open, close, 5)
    print("\n3. 测试表达式: open close corr delta_5")
    tokens = [
        TOKEN_DEFINITIONS['BEG'],
        TOKEN_DEFINITIONS['open'],
        TOKEN_DEFINITIONS['close'],
        TOKEN_DEFINITIONS['corr'],
        TOKEN_DEFINITIONS['delta_5'],
        TOKEN_DEFINITIONS['END']
    ]
    result = RPNEvaluator.evaluate(tokens, data_dict)
    print(f"   结果: {result.values}")

    # 测试表达式 4: 复杂表达式 log(volume) * sign(close - open)
    print("\n4. 测试复杂表达式: volume log close open sub sign mul")
    tokens = [
        TOKEN_DEFINITIONS['BEG'],
        TOKEN_DEFINITIONS['volume'],
        TOKEN_DEFINITIONS['log'],
        TOKEN_DEFINITIONS['close'],
        TOKEN_DEFINITIONS['open'],
        TOKEN_DEFINITIONS['sub'],
        TOKEN_DEFINITIONS['sign'],
        TOKEN_DEFINITIONS['mul'],
        TOKEN_DEFINITIONS['END']
    ]
    result = RPNEvaluator.evaluate(tokens, data_dict)
    print(f"   结果: {result.values}")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" " * 15 + "量化因子算子测试套件")
    print("=" * 60)

    test_unary_operators()
    test_binary_operators()
    test_time_series_operators()
    test_correlation_operators()
    test_rpn_evaluation()

    print("\n" + "=" * 60)
    print(" " * 20 + "测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()