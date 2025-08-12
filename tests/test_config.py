"""配置模块测试文件 - 测试所有配置项的有效性和一致性"""

import pytest
from config.config import (
    MCTS_CONFIG,
    RISK_SEEKING_CONFIG,
    ALPHA_POOL_CONFIG,
    GRU_CONFIG,
    POLICY_CONFIG,
    CV_CONFIG,
    DATA_CONFIG,
    BACKTEST_CONFIG,
    validate_config
)


class TestMCTSConfig:
    """测试MCTS配置"""

    def test_required_fields(self):
        """测试必需字段"""
        required_fields = [
            'num_iterations',
            'risk_seeking_exploration',
            'max_episode_length',
            'num_simulations',
            'gamma',
            'c_puct',
            'exploration_constant'
        ]

        for field in required_fields:
            assert field in MCTS_CONFIG
            assert MCTS_CONFIG[field] is not None

    def test_value_ranges(self):
        """测试值的合理范围"""
        # 迭代次数应该是正整数
        assert isinstance(MCTS_CONFIG['num_iterations'], int)
        assert MCTS_CONFIG['num_iterations'] > 0

        # gamma应该在[0, 1]之间
        assert 0 <= MCTS_CONFIG['gamma'] <= 1

        # 探索常数应该是正数
        assert MCTS_CONFIG['c_puct'] > 0
        assert MCTS_CONFIG['exploration_constant'] > 0

        # 最大episode长度应该合理
        assert 10 <= MCTS_CONFIG['max_episode_length'] <= 100

    def test_paper_requirements(self):
        """测试论文要求的配置"""
        # 论文要求gamma=1.0来鼓励长表达式
        assert MCTS_CONFIG['gamma'] == 1.0

        # 论文指定的迭代次数
        assert MCTS_CONFIG['num_iterations'] == 200
        assert MCTS_CONFIG['num_simulations'] == 200

    def test_exploration_decay(self):
        """测试探索衰减参数"""
        assert 'c_puct_decay' in MCTS_CONFIG
        assert 0 < MCTS_CONFIG['c_puct_decay'] <= 1

        assert 'c_puct_min' in MCTS_CONFIG
        assert MCTS_CONFIG['c_puct_min'] > 0
        assert MCTS_CONFIG['c_puct_min'] < MCTS_CONFIG['c_puct']

    def test_additional_parameters(self):
        """测试额外参数"""
        # 窗口大小限制
        if 'min_window_size' in MCTS_CONFIG:
            assert MCTS_CONFIG['min_window_size'] >= 1

        # 常数惩罚
        if 'constant_penalty' in MCTS_CONFIG:
            assert MCTS_CONFIG['constant_penalty'] <= 0

        # 多样性奖励
        if 'diversity_bonus' in MCTS_CONFIG:
            assert MCTS_CONFIG['diversity_bonus'] >= 0


class TestRiskSeekingConfig:
    """测试风险寻求配置"""

    def test_quantile_threshold(self):
        """测试分位数阈值"""
        assert 'quantile_threshold' in RISK_SEEKING_CONFIG
        assert 0 < RISK_SEEKING_CONFIG['quantile_threshold'] < 1

        # 论文提到优化top 15%
        assert RISK_SEEKING_CONFIG['quantile_threshold'] == 0.85

    def test_learning_rates(self):
        """测试学习率"""
        assert 'learning_rate_beta' in RISK_SEEKING_CONFIG
        assert 'learning_rate_gamma' in RISK_SEEKING_CONFIG

        # 学习率应该是小的正数
        assert 0 < RISK_SEEKING_CONFIG['learning_rate_beta'] < 1
        assert 0 < RISK_SEEKING_CONFIG['learning_rate_gamma'] < 1

        # beta通常比gamma大
        assert RISK_SEEKING_CONFIG['learning_rate_beta'] >= RISK_SEEKING_CONFIG['learning_rate_gamma']

    def test_gradient_clipping(self):
        """测试梯度裁剪"""
        assert 'gradient_clip' in RISK_SEEKING_CONFIG
        assert RISK_SEEKING_CONFIG['gradient_clip'] > 0
        assert RISK_SEEKING_CONFIG['gradient_clip'] <= 1.0


class TestAlphaPoolConfig:
    """测试Alpha池配置"""

    def test_paper_specifications(self):
        """测试论文规定的参数"""
        # 论文指定K=100
        assert ALPHA_POOL_CONFIG['pool_size'] == 100

        # 论文指定λ=0.1 (reward-dense MDP)
        assert ALPHA_POOL_CONFIG['lambda_param'] == 0.1

    def test_optimization_parameters(self):
        """测试优化参数"""
        assert 'gradient_descent_lr' in ALPHA_POOL_CONFIG
        assert 0 < ALPHA_POOL_CONFIG['gradient_descent_lr'] < 1

        assert 'gradient_descent_iters' in ALPHA_POOL_CONFIG
        assert ALPHA_POOL_CONFIG['gradient_descent_iters'] > 0

    def test_quality_thresholds(self):
        """测试质量阈值"""
        if 'min_std' in ALPHA_POOL_CONFIG:
            assert ALPHA_POOL_CONFIG['min_std'] > 0
            assert ALPHA_POOL_CONFIG['min_std'] < 0.01

        if 'min_unique_ratio' in ALPHA_POOL_CONFIG:
            assert 0 < ALPHA_POOL_CONFIG['min_unique_ratio'] < 1

        if 'min_ic_threshold' in ALPHA_POOL_CONFIG:
            assert ALPHA_POOL_CONFIG['min_ic_threshold'] > 0
            assert ALPHA_POOL_CONFIG['min_ic_threshold'] < 0.1


class TestNetworkConfig:
    """测试网络配置"""

    def test_gru_config(self):
        """测试GRU配置"""
        # 论文指定4层GRU
        assert GRU_CONFIG['num_layers'] == 4

        # 论文指定隐藏维度64
        assert GRU_CONFIG['hidden_dim'] == 64

    def test_policy_config(self):
        """测试策略网络配置"""
        # 层数和维度
        assert POLICY_CONFIG['gru_layers'] == 4
        assert POLICY_CONFIG['gru_hidden_dim'] == 64

        # MLP配置
        assert 'policy_hidden_layers' in POLICY_CONFIG
        assert len(POLICY_CONFIG['policy_hidden_layers']) == 2
        assert all(n == 32 for n in POLICY_CONFIG['policy_hidden_layers'])

        # Dropout率
        if 'dropout_rate' in POLICY_CONFIG:
            assert 0 <= POLICY_CONFIG['dropout_rate'] < 1

    def test_consistency(self):
        """测试配置一致性"""
        # GRU配置应该与Policy配置一致
        assert GRU_CONFIG['num_layers'] == POLICY_CONFIG['gru_layers']
        assert GRU_CONFIG['hidden_dim'] == POLICY_CONFIG['gru_hidden_dim']


class TestDataConfig:
    """测试数据配置"""

    def test_feature_specification(self):
        """测试特征规范"""
        assert 'features' in DATA_CONFIG

        expected_features = ['open', 'high', 'low', 'close', 'volume', 'vwap']
        for feature in expected_features:
            assert feature in DATA_CONFIG['features']

    def test_target_configuration(self):
        """测试目标配置"""
        assert 'target_column' in DATA_CONFIG
        assert isinstance(DATA_CONFIG['target_column'], str)

        if 'target_windows' in DATA_CONFIG:
            assert all(w > 0 for w in DATA_CONFIG['target_windows'])

    def test_period_specification(self):
        """测试时期规范"""
        periods = ['train_period', 'val_period', 'test_period']

        for period in periods:
            if period in DATA_CONFIG:
                assert isinstance(DATA_CONFIG[period], str)
                # 应该包含日期范围
                assert 'to' in DATA_CONFIG[period] or '-' in DATA_CONFIG[period]

    def test_data_path(self):
        """测试数据路径"""
        if 'default_data_path' in DATA_CONFIG:
            path = DATA_CONFIG['default_data_path']
            assert isinstance(path, str)
            # 路径应该有合理的扩展名
            assert path.endswith('.csv') or path.endswith('.pt') or path.endswith('.pkl')


class TestBacktestConfig:
    """测试回测配置"""

    def test_trading_parameters(self):
        """测试交易参数"""
        assert 'top_k' in BACKTEST_CONFIG
        assert BACKTEST_CONFIG['top_k'] > 0
        assert BACKTEST_CONFIG['top_k'] <= 100

        assert 'rebalance_freq' in BACKTEST_CONFIG
        assert BACKTEST_CONFIG['rebalance_freq'] > 0
        assert BACKTEST_CONFIG['rebalance_freq'] <= 30

    def test_cost_configuration(self):
        """测试成本配置"""
        assert 'transaction_cost' in BACKTEST_CONFIG
        assert 0 <= BACKTEST_CONFIG['transaction_cost'] < 0.01

        assert 'initial_capital' in BACKTEST_CONFIG
        assert BACKTEST_CONFIG['initial_capital'] > 0

    def test_paper_settings(self):
        """测试论文设置"""
        # 论文提到选择前40只股票
        assert BACKTEST_CONFIG['top_k'] == 40

        # 论文提到每5天重新平衡
        assert BACKTEST_CONFIG['rebalance_freq'] == 5


class TestCrossValidationConfig:
    """测试交叉验证配置"""

    def test_cv_splits(self):
        """测试CV分割数"""
        assert 'n_splits' in CV_CONFIG
        assert CV_CONFIG['n_splits'] > 1
        assert CV_CONFIG['n_splits'] <= 20

        # 论文使用8折
        assert CV_CONFIG['n_splits'] == 8


class TestConfigValidation:
    """测试配置验证函数"""

    def test_validate_function(self):
        """测试验证函数"""
        # 应该不抛出异常
        try:
            validate_config()
        except AssertionError as e:
            pytest.fail(f"Config validation failed: {e}")

    def test_validate_with_modification(self):
        """测试修改后的验证"""
        # 保存原始值
        original_gamma = MCTS_CONFIG['gamma']

        # 修改为无效值
        MCTS_CONFIG['gamma'] = 0.5

        # 应该验证失败
        with pytest.raises(AssertionError):
            validate_config()

        # 恢复
        MCTS_CONFIG['gamma'] = original_gamma

    def test_critical_parameters(self):
        """测试关键参数"""
        # 这些是论文中明确指定的参数
        critical_params = [
            (MCTS_CONFIG['gamma'], 1.0, "γ=1"),
            (ALPHA_POOL_CONFIG['lambda_param'], 0.1, "λ=0.1"),
            (ALPHA_POOL_CONFIG['pool_size'], 100, "K=100"),
            (GRU_CONFIG['num_layers'], 4, "4层GRU"),
            (GRU_CONFIG['hidden_dim'], 64, "隐藏维度64")
        ]

        for actual, expected, description in critical_params:
            assert actual == expected, f"配置不符合论文要求: {description}"


class TestConfigCompleteness:
    """测试配置完整性"""

    def test_all_modules_configured(self):
        """测试所有模块都有配置"""
        required_configs = [
            MCTS_CONFIG,
            RISK_SEEKING_CONFIG,
            ALPHA_POOL_CONFIG,
            GRU_CONFIG,
            POLICY_CONFIG,
            CV_CONFIG,
            DATA_CONFIG,
            BACKTEST_CONFIG
        ]

        for config in required_configs:
            assert config is not None
            assert len(config) > 0

    def test_no_missing_values(self):
        """测试没有缺失值"""
        all_configs = {
            'MCTS': MCTS_CONFIG,
            'RISK_SEEKING': RISK_SEEKING_CONFIG,
            'ALPHA_POOL': ALPHA_POOL_CONFIG,
            'GRU': GRU_CONFIG,
            'POLICY': POLICY_CONFIG,
            'CV': CV_CONFIG,
            'DATA': DATA_CONFIG,
            'BACKTEST': BACKTEST_CONFIG
        }

        for name, config in all_configs.items():
            for key, value in config.items():
                assert value is not None, f"{name}.{key} is None"

    def test_type_consistency(self):
        """测试类型一致性"""
        # 数值参数应该是数字
        numeric_params = [
            (MCTS_CONFIG['num_iterations'], int),
            (MCTS_CONFIG['gamma'], (int, float)),
            (ALPHA_POOL_CONFIG['lambda_param'], float),
            (BACKTEST_CONFIG['transaction_cost'], float)
        ]

        for value, expected_type in numeric_params:
            assert isinstance(value, expected_type)

    def test_interdependencies(self):
        """测试参数间的依赖关系"""
        # 探索衰减应该使最终值大于最小值
        final_c_puct = MCTS_CONFIG['c_puct'] * (
                MCTS_CONFIG['c_puct_decay'] ** MCTS_CONFIG['num_iterations']
        )
        assert final_c_puct >= MCTS_CONFIG['c_puct_min'] * 0.9  # 允许小误差

        # 学习率关系
        assert RISK_SEEKING_CONFIG['learning_rate_beta'] >= RISK_SEEKING_CONFIG['learning_rate_gamma']


class TestConfigUsability:
    """测试配置可用性"""

    def test_can_import(self):
        """测试可以导入所有配置"""
        from config.config import (
            MCTS_CONFIG,
            RISK_SEEKING_CONFIG,
            ALPHA_POOL_CONFIG,
            GRU_CONFIG,
            POLICY_CONFIG,
            CV_CONFIG,
            DATA_CONFIG,
            BACKTEST_CONFIG,
            validate_config
        )

        # 所有导入应该成功
        assert MCTS_CONFIG is not None
        assert callable(validate_config)

    def test_config_modification(self):
        """测试配置可修改性"""
        # 创建配置副本
        import copy
        config_copy = copy.deepcopy(MCTS_CONFIG)

        # 修改副本
        config_copy['num_iterations'] = 100

        # 原始配置不应该改变
        assert MCTS_CONFIG['num_iterations'] == 200
        assert config_copy['num_iterations'] == 100

    def test_config_serialization(self):
        """测试配置序列化"""
        import json

        # 应该可以序列化为JSON
        try:
            json_str = json.dumps(MCTS_CONFIG)
            loaded = json.loads(json_str)

            assert loaded['num_iterations'] == MCTS_CONFIG['num_iterations']
            assert loaded['gamma'] == MCTS_CONFIG['gamma']
        except (TypeError, ValueError) as e:
            pytest.fail(f"Config serialization failed: {e}")


class TestConfigDocumentation:
    """测试配置文档"""

    def test_parameter_comments(self):
        """测试参数注释（通过代码检查）"""
        # 这个测试主要是提醒维护者添加注释
        important_params = [
            'gamma',  # 折扣因子
            'lambda_param',  # 正则化参数
            'pool_size',  # Alpha池大小
            'quantile_threshold'  # 分位数阈值
        ]

        # 建议：每个重要参数都应该有注释说明其作用
        pass

    def test_example_usage(self):
        """测试示例用法"""
        # 演示如何使用配置
        from config.config import MCTS_CONFIG

        # 在MCTS中使用
        num_iterations = MCTS_CONFIG['num_iterations']
        assert num_iterations == 200

        # 在训练中使用
        gamma = MCTS_CONFIG['gamma']
        assert gamma == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])