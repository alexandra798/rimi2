"""验证模块测试文件 - 测试交叉验证和回测功能"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from validation.cross_validation import cross_validate_formulas, evaluate_formula_cross_val
from validation.backtest import backtest_formulas, backtest_with_trading_simulation
from alpha.evaluator import FormulaEvaluator


class TestCrossValidation:
    """测试交叉验证功能"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        n = 500

        # 创建有时间趋势的数据
        dates = pd.date_range('2020-01-01', periods=n)
        trend = np.linspace(100, 150, n)
        noise = np.random.randn(n) * 5

        X = pd.DataFrame({
            'close': trend + noise,
            'volume': np.random.exponential(1000000, n),
            'high': trend + noise + np.abs(np.random.randn(n) * 2),
            'low': trend + noise - np.abs(np.random.randn(n) * 2),
            'vwap': trend + noise * 0.5
        }, index=dates)

        # 创建相关的目标变量
        y = pd.Series(
            0.001 * (X['close'].values - 100) + np.random.randn(n) * 0.01,
            index=dates
        )

        return X, y

    @pytest.fixture
    def test_formulas(self):
        """测试用的公式"""
        return [
            "BEG close END",
            "BEG volume END",
            "BEG close volume add END",
            "BEG high low sub END"
        ]

    def test_evaluate_single_formula(self, sample_data, test_formulas):
        """测试评估单个公式"""
        X, y = sample_data
        formula = test_formulas[0]

        ic_scores = evaluate_formula_cross_val(
            formula, X, y,
            n_splits=3,
            evaluate_formula_func=None
        )

        assert isinstance(ic_scores, list)
        assert len(ic_scores) == 3  # n_splits

        # IC应该在[-1, 1]之间
        for ic in ic_scores:
            assert -1 <= ic <= 1

        # 不应该全是0（除非完全无相关）
        assert not all(ic == 0 for ic in ic_scores)

    def test_cross_validate_multiple_formulas(self, sample_data, test_formulas):
        """测试多个公式的交叉验证"""
        X, y = sample_data

        cv_results = cross_validate_formulas(
            test_formulas, X, y,
            n_splits=3,
            evaluate_formula_func=None
        )

        assert isinstance(cv_results, dict)
        assert len(cv_results) == len(test_formulas)

        for formula in test_formulas:
            assert formula in cv_results
            result = cv_results[formula]

            assert 'IC Scores' in result
            assert 'Mean IC' in result
            assert 'IC Std Dev' in result

            # 验证统计量
            assert result['Mean IC'] == pytest.approx(
                np.mean(result['IC Scores']), rel=1e-5
            )
            assert result['IC Std Dev'] == pytest.approx(
                np.std(result['IC Scores']), rel=1e-5
            )

    def test_time_series_split_ordering(self, sample_data):
        """测试时间序列分割的顺序性"""
        X, y = sample_data
        formula = "BEG close END"

        # 使用mock跟踪每个fold的数据
        fold_indices = []

        def mock_evaluate(formula, X_fold):
            fold_indices.append((X_fold.index[0], X_fold.index[-1]))
            return X_fold['close']

        with patch.object(FormulaEvaluator, 'evaluate', side_effect=mock_evaluate):
            evaluate_formula_cross_val(
                formula, X, y,
                n_splits=3,
                evaluate_formula_func=None
            )

        # 验证fold是按时间顺序的
        for i in range(len(fold_indices) - 1):
            # 每个fold的结束应该在下一个fold开始之前
            assert fold_indices[i][1] <= fold_indices[i + 1][0]

    def test_handle_nan_in_evaluation(self, sample_data):
        """测试处理评估中的NaN值"""
        X, y = sample_data

        # 添加一些NaN
        X_with_nan = X.copy()
        X_with_nan.iloc[0:10, 0] = np.nan
        y_with_nan = y.copy()
        y_with_nan.iloc[20:30] = np.nan

        formula = "BEG close END"

        # 不应该因为NaN而崩溃
        ic_scores = evaluate_formula_cross_val(
            formula, X_with_nan, y_with_nan,
            n_splits=3
        )

        assert len(ic_scores) == 3
        # 至少有一些非零IC（如果数据足够）
        assert any(ic != 0 for ic in ic_scores)

    def test_insufficient_data_handling(self):
        """测试数据不足的处理"""
        # 极少的数据
        X = pd.DataFrame({'close': [100, 101], 'volume': [1000, 1100]})
        y = pd.Series([0.01, 0.02])

        formula = "BEG close END"

        # 数据太少，分割后可能无法计算IC
        ic_scores = evaluate_formula_cross_val(
            formula, X, y,
            n_splits=2  # 只能分2折
        )

        # 应该返回结果，但可能都是0
        assert len(ic_scores) == 2
        # 数据太少，IC可能是0
        assert all(ic == 0 or abs(ic) <= 1 for ic in ic_scores)

    def test_custom_evaluator_function(self, sample_data):
        """测试自定义评估函数"""
        X, y = sample_data

        # 自定义评估函数
        def custom_evaluator(formula, data):
            # 简单返回close列
            return data['close']

        cv_results = cross_validate_formulas(
            ["BEG close END"],
            X, y,
            n_splits=3,
            evaluate_formula_func=custom_evaluator
        )

        assert len(cv_results) == 1
        # 应该使用自定义函数
        assert cv_results["BEG close END"]['Mean IC'] != 0


class TestBacktest:
    """测试回测功能"""

    @pytest.fixture
    def backtest_data(self):
        """创建回测数据"""
        np.random.seed(42)
        n = 200

        dates = pd.date_range('2023-01-01', periods=n)

        X = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(n) * 2),
            'volume': np.random.exponential(1000000, n),
            'high': 105 + np.cumsum(np.random.randn(n) * 2),
            'low': 95 + np.cumsum(np.random.randn(n) * 2),
            'vwap': 100 + np.cumsum(np.random.randn(n) * 1.5)
        }, index=dates)

        # 生成有轻微动量的收益率
        returns = np.random.randn(n) * 0.02
        for i in range(1, n):
            returns[i] += returns[i - 1] * 0.1  # 动量因子

        y = pd.Series(returns, index=dates)

        return X, y

    def test_simple_backtest(self, backtest_data):
        """测试简单回测"""
        X, y = backtest_data

        formulas = [
            "BEG close END",
            "BEG volume END",
            "BEG high low sub END"
        ]

        results = backtest_formulas(formulas, X, y)

        assert isinstance(results, dict)
        assert len(results) == len(formulas)

        for formula in formulas:
            assert formula in results
            ic = results[formula]

            # IC应该在合理范围内
            assert -1 <= ic <= 1
            assert isinstance(ic, (float, np.floating))

    def test_backtest_with_nan_handling(self, backtest_data):
        """测试回测中的NaN处理"""
        X, y = backtest_data

        # 添加NaN
        X_with_nan = X.copy()
        X_with_nan.iloc[0:20, 0] = np.nan

        formulas = ["BEG close END"]

        results = backtest_formulas(formulas, X_with_nan, y)

        # 应该返回结果
        assert len(results) == 1
        # IC可能较低但应该是有限值
        assert np.isfinite(results[formulas[0]])

    def test_formula_evaluation_error_handling(self, backtest_data):
        """测试公式评估错误处理"""
        X, y = backtest_data

        # 包含一个无效公式
        formulas = [
            "BEG close END",
            "BEG invalid_operation END",  # 无效操作
            "BEG volume END"
        ]

        with patch.object(FormulaEvaluator, 'evaluate') as mock_eval:
            # 模拟评估行为
            def side_effect(formula, data):
                if "invalid" in formula:
                    return pd.Series([np.nan] * len(data))
                elif "close" in formula:
                    return data['close']
                else:
                    return data['volume']

            mock_eval.side_effect = side_effect

            results = backtest_formulas(formulas, X, y)

            # 无效公式应该返回0或被处理
            assert formulas[1] in results
            assert results[formulas[1]] == 0

    @pytest.fixture
    def trading_data(self):
        """创建交易模拟数据"""
        np.random.seed(42)
        n_dates = 100
        n_stocks = 10

        dates = pd.date_range('2023-01-01', periods=n_dates)
        tickers = [f'STOCK_{i:03d}' for i in range(n_stocks)]

        # 创建MultiIndex数据
        index = pd.MultiIndex.from_product(
            [dates, tickers],
            names=['date', 'ticker']
        )

        # 生成价格数据
        base_prices = 100 + np.random.randn(n_stocks) * 20
        prices = []
        for stock_price in base_prices:
            stock_returns = np.random.randn(n_dates) * 0.02
            stock_prices = stock_price * np.exp(np.cumsum(stock_returns))
            prices.extend(stock_prices)

        X = pd.DataFrame({
            'close': prices,
            'volume': np.random.exponential(1000000, len(index)),
            'high': np.array(prices) * (1 + np.abs(np.random.randn(len(index)) * 0.01)),
            'low': np.array(prices) * (1 - np.abs(np.random.randn(len(index)) * 0.01))
        }, index=index)

        # 价格数据（用于交易模拟）
        price_data = X[['close']].copy()

        # 目标：未来收益率
        y = X.groupby('ticker')['close'].pct_change(5).shift(-5)

        return X, y, price_data

    def test_trading_simulation(self, trading_data):
        """测试交易模拟"""
        X, y, price_data = trading_data

        formulas = [
            "BEG close END",
            "BEG volume END"
        ]

        results = backtest_with_trading_simulation(
            formulas, X, y, price_data,
            top_k=5,
            rebalance_freq=10,
            initial_capital=100000
        )

        assert 'cumulative_return' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        assert 'portfolio_values' in results
        assert 'daily_returns' in results

        # 验证结果合理性
        assert isinstance(results['cumulative_return'], (float, np.floating))
        assert isinstance(results['sharpe_ratio'], (float, np.floating))
        assert results['max_drawdown'] >= 0  # 回撤应该是正数

        # 组合价值应该是递增的序列
        portfolio_values = results['portfolio_values']
        assert len(portfolio_values) > 0
        assert portfolio_values[0] == 100000  # 初始资金

    def test_trading_with_missing_prices(self, trading_data):
        """测试价格缺失时的交易模拟"""
        X, y, price_data = trading_data

        # 移除一些价格数据
        price_data_missing = price_data.copy()
        price_data_missing.iloc[100:150] = np.nan

        formulas = ["BEG close END"]

        # 应该能处理缺失价格
        results = backtest_with_trading_simulation(
            formulas, X, y, price_data_missing,
            top_k=3,
            rebalance_freq=20,
            initial_capital=100000
        )

        # 应该返回结果
        assert 'portfolio_values' in results
        # 组合价值不应该变成0
        assert all(v > 0 for v in results['portfolio_values'] if v is not None)

    def test_performance_metrics(self, trading_data):
        """测试性能指标计算"""
        X, y, price_data = trading_data

        # 创建简单的策略
        formulas = ["BEG close volume div END"]

        results = backtest_with_trading_simulation(
            formulas, X, y, price_data,
            top_k=5,
            rebalance_freq=5,
            initial_capital=100000
        )

        # Sharpe比率的合理范围
        assert -5 < results['sharpe_ratio'] < 5

        # 最大回撤应该在0-1之间（百分比）
        assert 0 <= results['max_drawdown'] <= 1

        # 累积收益率可能为负或正
        assert -1 < results['cumulative_return'] < 10  # 合理范围

        # 日收益率序列
        daily_returns = results['daily_returns']
        assert len(daily_returns) == len(results['portfolio_values']) - 1

        # 日收益率应该在合理范围
        assert all(-0.5 < r < 0.5 for r in daily_returns if not np.isnan(r))


class TestIntegrationValidation:
    """验证模块的集成测试"""

    def test_full_validation_pipeline(self):
        """测试完整的验证流程"""
        np.random.seed(42)

        # 生成数据
        n = 500
        X = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(n) * 2),
            'volume': np.random.exponential(1000000, n),
            'high': 105 + np.cumsum(np.random.randn(n) * 2),
            'low': 95 + np.cumsum(np.random.randn(n) * 2)
        })
        y = pd.Series(np.random.randn(n) * 0.01)

        # 定义策略
        formulas = [
            "BEG close END",
            "BEG volume END",
            "BEG high low sub END",
            "BEG close ts_mean delta_5 END"
        ]

        # 1. 交叉验证
        cv_results = cross_validate_formulas(
            formulas, X, y,
            n_splits=3
        )

        # 2. 选择最佳公式
        best_formula = max(
            cv_results.items(),
            key=lambda x: x[1]['Mean IC']
        )[0]

        assert best_formula in formulas

        # 3. 回测最佳公式
        backtest_results = backtest_formulas(
            [best_formula],
            X.iloc[-100:],  # 最后100个样本
            y.iloc[-100:]
        )

        assert best_formula in backtest_results

        # 4. 验证一致性
        # CV和回测的IC应该方向一致（都正或都负）
        cv_ic = cv_results[best_formula]['Mean IC']
        backtest_ic = backtest_results[best_formula]

        # 至少符号应该相同（或其中一个接近0）
        if abs(cv_ic) > 0.01 and abs(backtest_ic) > 0.01:
            assert np.sign(cv_ic) == np.sign(backtest_ic)

    def test_parallel_formula_evaluation(self):
        """测试并行公式评估（如果实现）"""
        # 这里可以测试多进程或多线程评估
        # 但原代码可能没有实现并行化
        pass

    def test_memory_efficiency(self):
        """测试大数据集的内存效率"""
        # 创建大数据集
        n = 10000
        X = pd.DataFrame({
            'close': np.random.randn(n),
            'volume': np.random.randn(n)
        })
        y = pd.Series(np.random.randn(n))

        formulas = ["BEG close END"]

        # 应该能处理大数据集
        cv_results = cross_validate_formulas(
            formulas, X, y,
            n_splits=2  # 少量分割以加快测试
        )

        assert len(cv_results) == 1
        assert not np.isnan(cv_results[formulas[0]]['Mean IC'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])