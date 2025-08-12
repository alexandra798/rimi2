"""Alpha池管理测试文件"""
import pytest
import numpy as np
import pandas as pd
from alpha.pool import AlphaPool
from alpha.evaluator import FormulaEvaluator


class TestAlphaPool:
    """测试Alpha池管理"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        n = 1000
        dates = pd.date_range('2020-01-01', periods=n)

        # 创建有相关性的数据
        trend = np.linspace(100, 150, n)
        noise = np.random.randn(n) * 5

        data = pd.DataFrame({
            'open': trend + noise,
            'high': trend + noise + np.abs(np.random.randn(n) * 2),
            'low': trend + noise - np.abs(np.random.randn(n) * 2),
            'close': trend + noise * 0.8,
            'volume': np.random.exponential(1000000, n),
            'vwap': trend + noise * 0.5
        }, index=dates)

        # 目标变量（与close相关）
        y = data['close'].shift(-5).fillna(method='ffill') / data['close'] - 1

        return data, y

    @pytest.fixture
    def alpha_pool(self):
        """创建Alpha池实例"""
        return AlphaPool(
            pool_size=10,
            lambda_param=0.1,
            learning_rate=0.01,
            min_std=1e-6,
            min_unique_ratio=0.01
        )

    def test_initialization(self, alpha_pool):
        """测试初始化"""
        assert alpha_pool.pool_size == 10
        assert alpha_pool.lambda_param == 0.1
        assert len(alpha_pool.alphas) == 0
        assert alpha_pool.rejected_constant_count == 0

    def test_is_valid_alpha(self, alpha_pool):
        """测试alpha有效性检查"""
        # 常数序列（无效）
        constant_values = pd.Series([1.0] * 100)
        assert alpha_pool.is_valid_alpha(constant_values) == False

        # 有变化的序列（有效）
        varying_values = pd.Series(np.random.randn(100))
        assert alpha_pool.is_valid_alpha(varying_values) == True

        # 太少有效值（无效）
        few_values = pd.Series([1, 2])
        assert alpha_pool.is_valid_alpha(few_values) == False

        # 包含NaN但有足够有效值
        with_nan = pd.Series([np.nan] * 50 + list(np.random.randn(50)))
        assert alpha_pool.is_valid_alpha(with_nan) == True

    def test_add_to_pool(self, alpha_pool):
        """测试添加alpha到池中"""
        # 添加有效alpha
        alpha_info = {
            'formula': 'BEG close volume add END',
            'score': 0.05,
            'ic': 0.05,
            'values': pd.Series(np.random.randn(100))
        }

        alpha_pool.add_to_pool(alpha_info)
        assert len(alpha_pool.alphas) == 1
        assert alpha_pool.alphas[0]['formula'] == alpha_info['formula']

        # 添加重复的alpha（应该被忽略）
        alpha_pool.add_to_pool(alpha_info)
        assert len(alpha_pool.alphas) == 1

        # 添加常数alpha（应该被拒绝）
        constant_alpha = {
            'formula': 'BEG const_1 END',
            'score': 0.01,
            'values': pd.Series([1.0] * 100)
        }

        alpha_pool.add_to_pool(constant_alpha)
        assert len(alpha_pool.alphas) == 1  # 仍然是1个
        assert alpha_pool.rejected_constant_count > 0

    def test_pool_size_limit(self, alpha_pool):
        """测试池大小限制"""
        # 添加超过池大小的alpha
        for i in range(15):
            alpha_info = {
                'formula': f'BEG close const_{i} mul END',
                'score': np.random.random() * 0.1,
                'ic': np.random.random() * 0.1,
                'values': pd.Series(np.random.randn(100) * (i + 1))
            }
            alpha_pool.add_to_pool(alpha_info)

        # 池大小不应超过限制
        assert len(alpha_pool.alphas) <= alpha_pool.pool_size

    def test_update_pool(self, alpha_pool, sample_data):
        """测试更新池"""
        X_data, y_data = sample_data
        evaluator = FormulaEvaluator()

        # 添加一些初始alpha
        formulas = [
            'BEG close END',
            'BEG volume END',
            'BEG close volume add END'
        ]

        for formula in formulas:
            alpha_pool.add_to_pool({
                'formula': formula,
                'score': 0.01
            })

        # 更新池
        alpha_pool.update_pool(X_data, y_data, evaluator)

        # 验证alpha被评估
        for alpha in alpha_pool.alphas:
            assert 'values' in alpha
            assert 'ic' in alpha
            assert alpha['values'] is not None

    def test_optimize_weights(self, alpha_pool, sample_data):
        """测试权重优化"""
        X_data, y_data = sample_data
        evaluator = FormulaEvaluator()

        # 创建有不同IC的alpha
        alpha1 = {
            'formula': 'BEG close END',
            'values': X_data['close'],
            'ic': 0.1,
            'weight': 0.5
        }

        alpha2 = {
            'formula': 'BEG volume END',
            'values': X_data['volume'],
            'ic': 0.05,
            'weight': 0.5
        }

        alpha_pool.alphas = [alpha1, alpha2]

        # 优化权重
        alpha_pool._optimize_weights_gradient_descent(X_data, y_data, max_iters=50)

        # 权重应该被更新
        assert alpha1['weight'] != 0.5
        assert alpha2['weight'] != 0.5

        # 权重之和不一定为1（没有归一化约束）
        # 但应该是有限值
        assert np.isfinite(alpha1['weight'])
        assert np.isfinite(alpha2['weight'])

    def test_get_top_formulas(self, alpha_pool):
        """测试获取最佳公式"""
        # 添加不同IC的alpha
        alphas = [
            {'formula': 'formula1', 'ic': 0.08, 'weight': 1.0},
            {'formula': 'formula2', 'ic': 0.12, 'weight': 1.0},
            {'formula': 'formula3', 'ic': 0.05, 'weight': 1.0},
            {'formula': 'formula4', 'ic': 0.15, 'weight': 1.0},
            {'formula': 'formula5', 'ic': 0.03, 'weight': 1.0}
        ]

        for alpha in alphas:
            alpha_pool.alphas.append(alpha)

        # 获取前3个
        top_formulas = alpha_pool.get_top_formulas(n=3)

        assert len(top_formulas) == 3
        # 应该按IC排序
        assert top_formulas[0] == 'formula4'  # IC=0.15
        assert top_formulas[1] == 'formula2'  # IC=0.12
        assert top_formulas[2] == 'formula1'  # IC=0.08

    def test_calculate_ic(self, alpha_pool):
        """测试IC计算"""
        # 完全相关
        pred = pd.Series([1, 2, 3, 4, 5])
        target = pd.Series([2, 4, 6, 8, 10])
        ic = alpha_pool._calculate_ic(pred, target)
        assert abs(ic - 1.0) < 0.01  # 应该接近1

        # 完全负相关
        target_neg = pd.Series([10, 8, 6, 4, 2])
        ic = alpha_pool._calculate_ic(pred, target_neg)
        assert abs(ic + 1.0) < 0.01  # 应该接近-1

        # 无相关
        target_random = pd.Series(np.random.randn(5))
        ic = alpha_pool._calculate_ic(pred, target_random)
        assert abs(ic) < 1.0  # 应该在[-1, 1]之间

        # 处理NaN
        pred_nan = pd.Series([1, np.nan, 3, 4, 5])
        target_nan = pd.Series([2, 4, np.nan, 8, 10])
        ic = alpha_pool._calculate_ic(pred_nan, target_nan)
        assert np.isfinite(ic)  # 应该返回有限值

    def test_composite_alpha_value(self, alpha_pool, sample_data):
        """测试合成alpha值计算"""
        X_data, y_data = sample_data

        # 添加带权重的alpha
        alpha1 = {
            'formula': 'formula1',
            'values': pd.Series(np.ones(len(X_data)), index=X_data.index),
            'weight': 0.3
        }

        alpha2 = {
            'formula': 'formula2',
            'values': pd.Series(np.ones(len(X_data)) * 2, index=X_data.index),
            'weight': 0.7
        }

        alpha_pool.alphas = [alpha1, alpha2]

        # 计算合成值
        composite = alpha_pool.get_composite_alpha_value(X_data)

        assert composite is not None
        # 加权平均：(1*0.3 + 2*0.7) / (0.3+0.7) = 1.7
        expected = (1 * 0.3 + 2 * 0.7) / (0.3 + 0.7)
        assert np.allclose(composite.values, expected)

    def test_pool_statistics(self, alpha_pool):
        """测试池统计信息"""
        # 空池
        stats = alpha_pool.get_pool_statistics()
        assert stats['pool_size'] == 0

        # 添加alpha
        alphas = [
            {'formula': 'f1', 'ic': 0.1, 'weight': 0.3},
            {'formula': 'f2', 'ic': 0.2, 'weight': 0.7},
            {'formula': 'f3', 'ic': -0.05, 'weight': 0.0}
        ]

        for alpha in alphas:
            alpha_pool.alphas.append(alpha)

        stats = alpha_pool.get_pool_statistics()

        assert stats['pool_size'] == 3
        assert np.isclose(stats['avg_ic'], np.mean([0.1, 0.2, -0.05]))
        assert stats['max_ic'] == 0.2
        assert stats['min_ic'] == -0.05
        assert np.isclose(stats['avg_weight'], np.mean([0.3, 0.7, 0.0]))

    def test_remove_worst_alpha(self, alpha_pool):
        """测试移除最差的alpha"""
        # 添加不同权重的alpha
        alphas = [
            {'formula': 'f1', 'weight': 0.5},
            {'formula': 'f2', 'weight': 0.01},  # 最小权重
            {'formula': 'f3', 'weight': 0.3}
        ]

        for alpha in alphas:
            alpha_pool.alphas.append(alpha)

        alpha_pool._remove_worst_alpha()

        assert len(alpha_pool.alphas) == 2
        # f2应该被移除
        formulas = [a['formula'] for a in alpha_pool.alphas]
        assert 'f2' not in formulas

    def test_edge_cases(self, alpha_pool):
        """测试边界情况"""
        # None值
        assert alpha_pool.is_valid_alpha(None) == False

        # 空Series
        assert alpha_pool.is_valid_alpha(pd.Series([])) == False

        # 全NaN
        all_nan = pd.Series([np.nan] * 100)
        assert alpha_pool.is_valid_alpha(all_nan) == False

        # 单一非零常数
        single_value = pd.Series([42.0] * 100)
        assert alpha_pool.is_valid_alpha(single_value) == False

        # 极小变化（接近常数）
        tiny_variation = pd.Series([1.0] * 99 + [1.0 + 1e-10])
        assert alpha_pool.is_valid_alpha(tiny_variation) == False


class TestFormulaEvaluator:
    """测试公式评估器"""

    @pytest.fixture
    def evaluator(self):
        return FormulaEvaluator()

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        return pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 900, 1200, 1050]
        })

    def test_evaluate_simple_formula(self, evaluator, sample_data):
        """测试简单公式评估"""
        formula = "BEG close END"
        result = evaluator.evaluate(formula, sample_data)

        assert result is not None
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert np.allclose(result.values, sample_data['close'].values)

    def test_evaluate_complex_formula(self, evaluator, sample_data):
        """测试复杂公式评估"""
        formula = "BEG close volume add const_2 div END"
        result = evaluator.evaluate(formula, sample_data)

        expected = (sample_data['close'] + sample_data['volume']) / 2

        assert result is not None
        assert np.allclose(result.values, expected.values)

    def test_evaluate_invalid_formula(self, evaluator, sample_data):
        """测试无效公式"""
        # 语法错误的公式
        formula = "BEG add END"  # add需要2个操作数
        result = evaluator.evaluate(formula, sample_data)

        # 应该返回NaN Series
        assert result is not None
        assert result.isna().all()

    def test_cache_mechanism(self, evaluator, sample_data):
        """测试缓存机制"""
        formula = "BEG close volume add END"

        # 第一次评估
        result1 = evaluator.evaluate(formula, sample_data)

        # 第二次评估（应该使用缓存）
        result2 = evaluator.evaluate(formula, sample_data)

        # 结果应该相同
        assert np.allclose(result1.values, result2.values)

        # 缓存应该有条目
        assert len(evaluator._result_cache) > 0

    def test_constant_detection(self, evaluator):
        """测试常数检测"""
        # 创建会产生常数的公式
        data = pd.DataFrame({'close': [100] * 100})

        formula = "BEG close ts_std delta_5 END"  # 常数的标准差是0
        result = evaluator.evaluate(formula, data)

        # 应该检测到常数并返回NaN
        assert result is not None
        # 结果应该被处理（填充或返回NaN）
        # 根据代码逻辑，常数会被转换为NaN
        assert result.isna().all() or (result == 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])