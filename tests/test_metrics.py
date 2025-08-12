"""评估指标测试文件 - 测试各种性能指标的计算"""

import pytest
import numpy as np
import pandas as pd
from utils.metrics import (
    calculate_ic,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_icir,
    calculate_rank_ic
)


class TestICCalculation:
    """测试信息系数(IC)计算"""

    def test_perfect_positive_correlation(self):
        """测试完全正相关"""
        pred = pd.Series([1, 2, 3, 4, 5])
        target = pd.Series([2, 4, 6, 8, 10])

        ic = calculate_ic(pred, target, method='pearson')
        assert abs(ic - 1.0) < 1e-6

        # Spearman也应该是1
        ic_spearman = calculate_ic(pred, target, method='spearman')
        assert abs(ic_spearman - 1.0) < 1e-6

    def test_perfect_negative_correlation(self):
        """测试完全负相关"""
        pred = pd.Series([1, 2, 3, 4, 5])
        target = pd.Series([10, 8, 6, 4, 2])

        ic = calculate_ic(pred, target, method='pearson')
        assert abs(ic + 1.0) < 1e-6

    def test_no_correlation(self):
        """测试无相关"""
        np.random.seed(42)
        pred = pd.Series(np.random.randn(100))
        target = pd.Series(np.random.randn(100))

        ic = calculate_ic(pred, target)
        # 随机数据IC应该接近0
        assert abs(ic) < 0.3

    def test_with_nan_values(self):
        """测试包含NaN值"""
        pred = pd.Series([1, np.nan, 3, 4, 5])
        target = pd.Series([2, 4, np.nan, 8, 10])

        ic = calculate_ic(pred, target)

        # 应该返回有限值
        assert np.isfinite(ic)
        # 使用剩余的有效值计算
        assert -1 <= ic <= 1

    def test_all_nan(self):
        """测试全NaN"""
        pred = pd.Series([np.nan, np.nan, np.nan])
        target = pd.Series([np.nan, np.nan, np.nan])

        ic = calculate_ic(pred, target)
        assert ic == 0.0

    def test_constant_values(self):
        """测试常数值"""
        pred = pd.Series([1, 1, 1, 1, 1])
        target = pd.Series([1, 2, 3, 4, 5])

        ic = calculate_ic(pred, target)
        # 常数预测应该返回0
        assert ic == 0.0

    def test_index_alignment(self):
        """测试索引对齐"""
        # 不同索引的Series
        pred = pd.Series([1, 2, 3, 4, 5], index=[0, 1, 2, 3, 4])
        target = pd.Series([2, 4, 6, 8, 10], index=[1, 2, 3, 4, 5])

        ic = calculate_ic(pred, target)

        # 应该基于交集计算
        assert np.isfinite(ic)
        assert -1 <= ic <= 1

    def test_array_input(self):
        """测试数组输入"""
        pred = np.array([1, 2, 3, 4, 5])
        target = np.array([2, 4, 6, 8, 10])

        ic = calculate_ic(pred, target)
        assert abs(ic - 1.0) < 1e-6

    def test_different_lengths(self):
        """测试不同长度"""
        pred = np.array([1, 2, 3, 4, 5])
        target = np.array([2, 4, 6])

        ic = calculate_ic(pred, target)

        # 应该截断到最短长度
        assert np.isfinite(ic)
        assert -1 <= ic <= 1

    def test_spearman_vs_pearson(self):
        """测试Spearman vs Pearson"""
        # 创建非线性但单调的关系
        x = np.array([1, 2, 3, 4, 5])
        y = x ** 2  # 二次关系

        ic_pearson = calculate_ic(x, y, method='pearson')
        ic_spearman = calculate_ic(x, y, method='spearman')

        # Spearman应该是1（完全单调）
        assert abs(ic_spearman - 1.0) < 1e-6
        # Pearson应该小于1（非线性）
        assert ic_pearson < 1.0
        assert ic_pearson > 0.9  # 但仍然很高


class TestSharpeRatio:
    """测试夏普比率计算"""

    def test_positive_sharpe(self):
        """测试正夏普比率"""
        # 稳定的正收益
        returns = pd.Series([0.01, 0.02, 0.01, 0.015, 0.01] * 50)

        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods=252)

        assert sharpe > 0
        # 合理的夏普比率范围
        assert 0 < sharpe < 10

    def test_negative_sharpe(self):
        """测试负夏普比率"""
        # 稳定的负收益
        returns = pd.Series([-0.01, -0.02, -0.01, -0.015, -0.01] * 50)

        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods=252)

        assert sharpe < 0

    def test_zero_volatility(self):
        """测试零波动率"""
        # 常数收益
        returns = pd.Series([0.01] * 100)

        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods=252)

        # 零波动率应该返回0或特殊值
        assert sharpe == 0.0

    def test_with_risk_free_rate(self):
        """测试带无风险利率"""
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.001)

        sharpe_no_rf = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        sharpe_with_rf = calculate_sharpe_ratio(returns, risk_free_rate=0.02)

        # 有无风险利率时夏普比率应该不同
        assert sharpe_no_rf != sharpe_with_rf

    def test_different_periods(self):
        """测试不同周期"""
        returns = pd.Series(np.random.randn(100) * 0.01 + 0.001)

        sharpe_daily = calculate_sharpe_ratio(returns, periods=252)
        sharpe_monthly = calculate_sharpe_ratio(returns, periods=12)

        # 不同周期的夏普比率应该不同
        assert sharpe_daily != sharpe_monthly

    def test_with_nan_values(self):
        """测试包含NaN值"""
        returns = pd.Series([0.01, np.nan, 0.02, 0.01, np.nan] * 20)

        sharpe = calculate_sharpe_ratio(returns)

        # 应该能处理NaN
        assert np.isfinite(sharpe)

    def test_single_return(self):
        """测试单个收益值"""
        returns = pd.Series([0.01])

        sharpe = calculate_sharpe_ratio(returns)

        # 单个值无法计算波动率
        assert sharpe == 0.0

    def test_extreme_values(self):
        """测试极端值"""
        returns = pd.Series([0.01] * 99 + [10.0])  # 一个极端值

        sharpe = calculate_sharpe_ratio(returns)

        # 应该能处理极端值
        assert np.isfinite(sharpe)


class TestMaxDrawdown:
    """测试最大回撤计算"""

    def test_no_drawdown(self):
        """测试无回撤"""
        # 单调递增
        cumulative_returns = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])

        mdd = calculate_max_drawdown(cumulative_returns)

        assert mdd == 0.0

    def test_simple_drawdown(self):
        """测试简单回撤"""
        # 先涨后跌
        cumulative_returns = np.array([1.0, 1.2, 1.1, 0.9, 1.0, 1.1])

        mdd = calculate_max_drawdown(cumulative_returns)

        # 从1.2跌到0.9，回撤25%
        expected = (1.2 - 0.9) / 1.2
        assert abs(mdd - expected) < 1e-6

    def test_multiple_drawdowns(self):
        """测试多次回撤"""
        cumulative_returns = np.array([
            1.0, 1.5, 1.2,  # 第一次回撤: 20%
            1.8, 1.0,  # 第二次回撤: 44.4%
            1.3, 1.1  # 第三次回撤: 15.4%
        ])

        mdd = calculate_max_drawdown(cumulative_returns)

        # 最大回撤是第二次
        expected = (1.8 - 1.0) / 1.8
        assert abs(mdd - expected) < 0.01

    def test_recovery(self):
        """测试恢复后的回撤"""
        cumulative_returns = np.array([
            1.0, 1.5, 1.0,  # 回撤33.3%
            2.0, 1.8, 2.5  # 新高后小回撤
        ])

        mdd = calculate_max_drawdown(cumulative_returns)

        # 最大回撤是第一次
        expected = (1.5 - 1.0) / 1.5
        assert abs(mdd - expected) < 0.01

    def test_zero_start(self):
        """测试从零开始"""
        cumulative_returns = np.array([0.0, 0.5, 1.0, 0.8, 1.2])

        mdd = calculate_max_drawdown(cumulative_returns)

        # 应该能处理零值
        assert np.isfinite(mdd)
        assert mdd >= 0

    def test_negative_returns(self):
        """测试负收益"""
        cumulative_returns = np.array([1.0, 0.8, 0.6, 0.4, 0.2])

        mdd = calculate_max_drawdown(cumulative_returns)

        # 从1.0跌到0.2，回撤80%
        expected = 0.8
        assert abs(mdd - expected) < 1e-6

    def test_pandas_series(self):
        """测试Pandas Series输入"""
        cumulative_returns = pd.Series([1.0, 1.2, 0.9, 1.1, 1.3])

        mdd = calculate_max_drawdown(cumulative_returns)

        expected = (1.2 - 0.9) / 1.2
        assert abs(mdd - expected) < 1e-6


class TestICIR:
    """测试IC Information Ratio"""

    def test_stable_ic(self):
        """测试稳定的IC"""
        # 稳定的IC序列
        ic_series = [0.05, 0.04, 0.06, 0.05, 0.04, 0.05] * 10

        icir = calculate_icir(ic_series)

        # ICIR应该很高（稳定）
        assert icir > 1.0

        # 计算验证
        mean_ic = np.mean(ic_series)
        std_ic = np.std(ic_series)
        expected = mean_ic / std_ic
        assert abs(icir - expected) < 1e-6

    def test_volatile_ic(self):
        """测试波动的IC"""
        # 波动大的IC序列
        ic_series = [0.1, -0.1, 0.2, -0.15, 0.05, -0.05] * 10

        icir = calculate_icir(ic_series)

        # ICIR应该较低（不稳定）
        assert abs(icir) < 1.0

    def test_zero_std(self):
        """测试零标准差"""
        # 常数IC
        ic_series = [0.05] * 10

        icir = calculate_icir(ic_series)

        # 标准差为0时的处理
        assert icir == 0.0 or np.isinf(icir)

    def test_negative_ic(self):
        """测试负IC"""
        ic_series = [-0.05, -0.04, -0.06, -0.05] * 10

        icir = calculate_icir(ic_series)

        # 负IC的ICIR应该是负的
        assert icir < 0

    def test_with_nan(self):
        """测试包含NaN"""
        ic_series = [0.05, np.nan, 0.04, 0.06, np.nan, 0.05] * 5

        icir = calculate_icir(ic_series)

        # 应该能处理NaN
        assert np.isfinite(icir)

    def test_insufficient_data(self):
        """测试数据不足"""
        ic_series = [0.05]  # 只有一个值

        icir = calculate_icir(ic_series)

        # 数据不足时返回0
        assert icir == 0.0


class TestRankIC:
    """测试Rank IC计算"""

    def test_perfect_rank_correlation(self):
        """测试完美的排序相关"""
        pred = pd.Series([1, 2, 3, 4, 5])
        target = pd.Series([10, 20, 30, 40, 50])

        rank_ic = calculate_rank_ic(pred, target)

        # 排序完全一致
        assert abs(rank_ic - 1.0) < 1e-6

    def test_inverse_rank_correlation(self):
        """测试反向排序相关"""
        pred = pd.Series([1, 2, 3, 4, 5])
        target = pd.Series([50, 40, 30, 20, 10])

        rank_ic = calculate_rank_ic(pred, target)

        # 排序完全相反
        assert abs(rank_ic + 1.0) < 1e-6

    def test_nonlinear_relationship(self):
        """测试非线性关系"""
        # 非线性但单调
        pred = np.array([1, 2, 3, 4, 5])
        target = pred ** 3  # 立方关系

        rank_ic = calculate_rank_ic(pred, target)

        # Rank IC应该是1（完全单调）
        assert abs(rank_ic - 1.0) < 1e-6

    def test_ties_handling(self):
        """测试相同值的处理"""
        pred = pd.Series([1, 1, 2, 2, 3])
        target = pd.Series([10, 10, 20, 20, 30])

        rank_ic = calculate_rank_ic(pred, target)

        # 应该能处理相同值
        assert np.isfinite(rank_ic)
        assert -1 <= rank_ic <= 1

    def test_outlier_robustness(self):
        """测试对异常值的鲁棒性"""
        pred = pd.Series([1, 2, 3, 4, 1000])  # 一个异常值
        target = pd.Series([10, 20, 30, 40, 50])

        rank_ic = calculate_rank_ic(pred, target)

        # Rank IC应该对异常值更鲁棒
        assert rank_ic > 0.5  # 仍然应该有正相关

    def test_with_nan(self):
        """测试包含NaN"""
        pred = pd.Series([1, np.nan, 3, 4, 5])
        target = pd.Series([10, 20, np.nan, 40, 50])

        rank_ic = calculate_rank_ic(pred, target)

        # 应该能处理NaN
        assert np.isfinite(rank_ic)
        assert -1 <= rank_ic <= 1


class TestEdgeCases:
    """测试边界情况"""

    def test_empty_input(self):
        """测试空输入"""
        empty_series = pd.Series([])

        ic = calculate_ic(empty_series, empty_series)
        assert ic == 0.0

        sharpe = calculate_sharpe_ratio(empty_series)
        assert sharpe == 0.0

        mdd = calculate_max_drawdown(empty_series)
        assert mdd == 0.0

    def test_single_value(self):
        """测试单个值"""
        single = pd.Series([1.0])

        ic = calculate_ic(single, single)
        assert ic == 0.0  # 无法计算相关性

        sharpe = calculate_sharpe_ratio(single)
        assert sharpe == 0.0  # 无法计算波动率

    def test_inf_values(self):
        """测试无穷值"""
        with_inf = pd.Series([1, 2, np.inf, 4, 5])
        normal = pd.Series([1, 2, 3, 4, 5])

        ic = calculate_ic(with_inf, normal)
        # 应该处理inf（可能忽略或返回特定值）
        assert np.isfinite(ic) or ic == 0.0

    def test_very_small_values(self):
        """测试极小值"""
        tiny = pd.Series([1e-15, 2e-15, 3e-15])

        ic = calculate_ic(tiny, tiny * 2)
        # 应该能处理极小值
        assert abs(ic - 1.0) < 0.01 or ic == 0.0  # 可能因精度问题返回0

    def test_very_large_values(self):
        """测试极大值"""
        huge = pd.Series([1e15, 2e15, 3e15])

        sharpe = calculate_sharpe_ratio(huge)
        # 应该能处理极大值
        assert np.isfinite(sharpe) or sharpe == 0.0


class TestMetricsConsistency:
    """测试指标之间的一致性"""

    def test_ic_and_rank_ic_consistency(self):
        """测试IC和Rank IC的一致性"""
        np.random.seed(42)

        # 线性关系
        pred = pd.Series(np.random.randn(100))
        target = pred * 2 + np.random.randn(100) * 0.1

        ic = calculate_ic(pred, target)
        rank_ic = calculate_rank_ic(pred, target)

        # 对于近似线性关系，两者应该相近
        assert abs(ic - rank_ic) < 0.1

        # 符号应该一致
        assert np.sign(ic) == np.sign(rank_ic)

    def test_metrics_on_random_data(self):
        """测试随机数据上的指标"""
        np.random.seed(42)

        # 完全随机
        random_pred = pd.Series(np.random.randn(1000))
        random_target = pd.Series(np.random.randn(1000))

        ic = calculate_ic(random_pred, random_target)
        rank_ic = calculate_rank_ic(random_pred, random_target)

        # 随机数据的IC应该接近0
        assert abs(ic) < 0.1
        assert abs(rank_ic) < 0.1

        # 随机收益的夏普比率应该接近0
        sharpe = calculate_sharpe_ratio(random_pred * 0.01)
        assert abs(sharpe) < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])