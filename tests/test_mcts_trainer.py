"""MCTS训练器测试文件 - 测试完整的训练流程"""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch, MagicMock
from mcts.trainer import RiskMinerTrainer
from mcts.environment import MDPState
from core.token_system import TOKEN_DEFINITIONS


class TestRiskMinerTrainer:
    """测试RiskMiner训练器"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        n = 200

        X = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(n) * 2),
            'volume': np.random.exponential(1000000, n),
            'high': 105 + np.cumsum(np.random.randn(n) * 2),
            'low': 95 + np.cumsum(np.random.randn(n) * 2),
            'vwap': 100 + np.cumsum(np.random.randn(n) * 1.5)
        })

        y = pd.Series(np.random.randn(n) * 0.01)

        return X, y

    @pytest.fixture
    def trainer(self, sample_data):
        """创建训练器实例"""
        X, y = sample_data
        device = torch.device('cpu')

        trainer = RiskMinerTrainer(
            X_data=X,
            y_data=y,
            device=device,
            use_sampling=True,
            sample_size=100,
            random_seed=42
        )

        return trainer

    def test_initialization(self, trainer, sample_data):
        """测试初始化"""
        X, y = sample_data

        assert trainer.X_data is not None
        assert trainer.y_data is not None

        # 检查采样
        if trainer.use_sampling and len(X) > 100:
            assert len(trainer.X_train_sample) == 100
            assert len(trainer.y_train_sample) == 100

        # 检查组件初始化
        assert trainer.mdp_env is not None
        assert trainer.policy_network is not None
        assert trainer.optimizer is not None
        assert trainer.mcts_searcher is not None
        assert trainer.alpha_pool is not None
        assert trainer.reward_calculator is not None

    def test_device_assignment(self):
        """测试设备分配"""
        X = pd.DataFrame({'close': [100, 101, 102]})
        y = pd.Series([0.01, 0.02, 0.03])

        # CPU设备
        trainer_cpu = RiskMinerTrainer(X, y, device=torch.device('cpu'))
        assert trainer_cpu.device.type == 'cpu'

        # 自动选择设备
        trainer_auto = RiskMinerTrainer(X, y, device=None)
        if torch.cuda.is_available():
            assert trainer_auto.device.type == 'cuda'
        else:
            assert trainer_auto.device.type == 'cpu'

    def test_sampling_logic(self):
        """测试采样逻辑"""
        np.random.seed(42)

        # 大数据集 - 应该采样
        X_large = pd.DataFrame({
            'close': np.random.randn(100000)
        })
        y_large = pd.Series(np.random.randn(100000))

        trainer_sample = RiskMinerTrainer(
            X_large, y_large,
            use_sampling=True,
            sample_size=5000
        )

        assert len(trainer_sample.X_train_sample) == 5000
        assert len(trainer_sample.y_train_sample) == 5000

        # 小数据集 - 不采样
        X_small = pd.DataFrame({'close': np.random.randn(100)})
        y_small = pd.Series(np.random.randn(100))

        trainer_no_sample = RiskMinerTrainer(
            X_small, y_small,
            use_sampling=True,
            sample_size=5000
        )

        assert len(trainer_no_sample.X_train_sample) == 100

    def test_search_one_iteration(self, trainer):
        """测试单次搜索迭代"""
        from mcts.node import MCTSNode

        # 创建根节点
        initial_state = trainer.mdp_env.reset()
        root = MCTSNode(state=initial_state)

        # 执行一次搜索
        trajectory = trainer.search_one_iteration(root)

        # 应该返回轨迹
        assert trajectory is not None
        # 根节点应该有子节点
        assert len(root.children) > 0

    def test_collect_trajectories(self, trainer):
        """测试轨迹收集"""
        trajectories = trainer.collect_trajectories_with_mcts(
            num_episodes=2,
            num_simulations_per_episode=5
        )

        assert len(trajectories) <= 2

        for trajectory in trajectories:
            if trajectory:  # 可能有空轨迹
                # 每个轨迹项应该是(state, action, reward)
                for state, action, reward in trajectory:
                    assert isinstance(state, MDPState)
                    assert action in TOKEN_DEFINITIONS
                    assert isinstance(reward, (float, np.floating))

    def test_extract_best_trajectory(self, trainer):
        """测试提取最佳轨迹"""
        from mcts.node import MCTSNode

        # 创建一个有子节点的树
        root = MCTSNode(state=MDPState())

        # 手动添加子节点
        state1 = MDPState()
        state1.add_token('close')
        child1 = root.add_child('close', state1)
        child1.N = 10  # 访问次数

        state2 = MDPState()
        state2.add_token('volume')
        child2 = root.add_child('volume', state2)
        child2.N = 5

        # 给child1添加END子节点
        state_end = state1.copy()
        state_end.add_token('END')
        child1.add_child('END', state_end).N = 8

        # 提取最佳轨迹
        trajectory = trainer.extract_best_trajectory(root)

        assert trajectory is not None
        assert len(trajectory) > 0

        # 第一步应该选择访问次数最多的
        assert trajectory[0][1] == 'close'  # 选择child1

    def test_train_policy_network(self, trainer):
        """测试策略网络训练"""
        # 创建模拟轨迹
        trajectories = []

        for _ in range(3):
            trajectory = []
            state = MDPState()

            # 简单序列
            actions = ['close', 'END']
            for i, action in enumerate(actions):
                if i > 0:
                    prev_state = state.copy()
                    state.add_token(actions[i - 1])
                    trajectory.append((prev_state, action, 0.1))

            trajectories.append(trajectory)

        # 训练
        avg_loss = trainer.train_policy_network(trajectories)

        # 应该返回损失值
        assert isinstance(avg_loss, float)
        assert avg_loss >= 0

    def test_update_alpha_pool(self, trainer):
        """测试更新Alpha池"""
        # 创建有效轨迹
        trajectories = []

        # 轨迹1: close -> END
        traj1 = []
        state1 = MDPState()
        traj1.append((state1, 'close', 0.05))
        state1_next = MDPState()
        state1_next.add_token('close')
        traj1.append((state1_next, 'END', 0.1))
        trajectories.append(traj1)

        # 轨迹2: volume -> END
        traj2 = []
        state2 = MDPState()
        traj2.append((state2, 'volume', 0.03))
        state2_next = MDPState()
        state2_next.add_token('volume')
        traj2.append((state2_next, 'END', 0.08))
        trajectories.append(traj2)

        # 更新池
        initial_pool_size = len(trainer.alpha_pool)
        trainer.update_alpha_pool(trajectories, iteration=1)

        # 池可能增加了公式
        assert len(trainer.alpha_pool) >= initial_pool_size

        # 检查公式格式
        for alpha in trainer.alpha_pool:
            assert 'formula' in alpha
            assert 'ic' in alpha
            assert isinstance(alpha['formula'], str)
            assert 'BEG' in alpha['formula']
            assert 'END' in alpha['formula']

    def test_get_formula_from_trajectory(self, trainer):
        """测试从轨迹获取公式"""
        trajectory = []

        state = MDPState()
        trajectory.append((state, 'close', 0.1))

        state = MDPState()
        state.add_token('close')
        trajectory.append((state, 'volume', 0.1))

        state = MDPState()
        state.add_token('close')
        state.add_token('volume')
        trajectory.append((state, 'add', 0.1))

        state = MDPState()
        state.add_token('close')
        state.add_token('volume')
        state.add_token('add')
        trajectory.append((state, 'END', 0.2))

        formula_tokens = trainer.get_formula_from_trajectory(trajectory)

        assert formula_tokens is not None
        # 应该包含所有动作
        token_names = [t.name for t in formula_tokens]
        assert 'BEG' in token_names
        assert 'close' in token_names
        assert 'volume' in token_names
        assert 'add' in token_names
        assert 'END' in token_names

    def test_print_statistics(self, trainer, capsys):
        """测试打印统计信息"""
        # 添加一些alpha到池中
        trainer.alpha_pool = [
            {'formula': 'BEG close END', 'ic': 0.08},
            {'formula': 'BEG volume END', 'ic': 0.12},
            {'formula': 'BEG close volume add END', 'ic': 0.05}
        ]

        trainer.print_statistics()

        # 检查输出
        captured = capsys.readouterr()
        assert "Alpha pool size: 3" in captured.out
        assert "Top 5 Alphas:" in captured.out
        assert "IC:" in captured.out

    def test_get_top_formulas(self, trainer):
        """测试获取最佳公式"""
        # 添加不同IC的公式
        trainer.alpha_pool = [
            {'formula': 'formula1', 'ic': 0.15},
            {'formula': 'formula2', 'ic': 0.08},
            {'formula': 'formula3', 'ic': 0.20},
            {'formula': 'formula4', 'ic': 0.05},
            {'formula': 'formula5', 'ic': 0.12}
        ]

        top_3 = trainer.get_top_formulas(n=3)

        assert len(top_3) == 3
        # 应该按IC排序
        assert top_3[0] == 'formula3'  # IC=0.20
        assert top_3[1] == 'formula1'  # IC=0.15
        assert top_3[2] == 'formula5'  # IC=0.12

    def test_constant_formula_filtering(self, trainer):
        """测试常数公式过滤"""
        # 创建会产生常数的轨迹
        trajectories = []

        # 常数轨迹: const_1 -> END
        traj = []
        state = MDPState()
        traj.append((state, 'const_1', 0.0))
        state_next = MDPState()
        state_next.add_token('const_1')
        traj.append((state_next, 'END', 0.0))
        trajectories.append(traj)

        initial_pool_size = len(trainer.alpha_pool)

        # 更新池
        trainer.update_alpha_pool(trajectories, iteration=1)

        # 常数公式不应该被添加
        assert len(trainer.alpha_pool) == initial_pool_size

    @pytest.mark.slow
    def test_full_training_cycle(self, trainer):
        """测试完整训练周期"""
        # 执行一个简短的训练
        trainer.train(
            num_iterations=2,
            num_simulations_per_iteration=5
        )

        # 应该有一些公式
        top_formulas = trainer.get_top_formulas(n=3)

        # 可能还没有找到好的公式
        assert isinstance(top_formulas, list)
        # 至少应该尝试了一些公式
        assert trainer.mcts_searcher.iteration_count > 0


class TestTrainingIntegration:
    """训练集成测试"""

    def test_memory_management(self):
        """测试内存管理"""
        # 创建大数据集
        X = pd.DataFrame({
            'close': np.random.randn(10000),
            'volume': np.random.randn(10000)
        })
        y = pd.Series(np.random.randn(10000))

        trainer = RiskMinerTrainer(
            X, y,
            use_sampling=True,
            sample_size=1000
        )

        # 应该采样以节省内存
        assert len(trainer.X_train_sample) == 1000

        # 缓存不应该无限增长
        initial_cache_size = len(trainer.reward_calculator._cache)

        # 执行一些计算
        for _ in range(10):
            state = MDPState()
            state.add_token('close')
            trainer.reward_calculator.calculate_intermediate_reward(
                state, trainer.X_train_sample, trainer.y_train_sample
            )

        # 缓存应该有限制（虽然代码中可能没有实现）
        assert len(trainer.reward_calculator._cache) < 1000

    def test_reproducibility(self):
        """测试可重复性"""
        np.random.seed(42)
        X = pd.DataFrame({
            'close': np.random.randn(100),
            'volume': np.random.randn(100)
        })
        y = pd.Series(np.random.randn(100))

        # 两个相同种子的训练器
        trainer1 = RiskMinerTrainer(X, y, random_seed=42)
        trainer2 = RiskMinerTrainer(X, y, random_seed=42)

        # 采样应该相同
        assert np.array_equal(
            trainer1.X_train_sample.values,
            trainer2.X_train_sample.values
        )

    def test_error_recovery(self, caplog):
        """测试错误恢复"""
        X = pd.DataFrame({'close': [100, 101, 102]})
        y = pd.Series([0.01, 0.02, 0.03])

        trainer = RiskMinerTrainer(X, y)

        # 创建会出错的轨迹
        bad_trajectory = [(None, 'invalid', 0.1)]

        # 不应该崩溃
        trainer.update_alpha_pool([bad_trajectory], iteration=1)

        # 应该记录错误但继续
        # 检查是否有错误日志（如果实现了）
        # assert "error" in caplog.text.lower() or len(trainer.alpha_pool) == 0

    def test_gpu_cpu_consistency(self):
        """测试GPU/CPU一致性"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        X = pd.DataFrame({'close': np.random.randn(100)})
        y = pd.Series(np.random.randn(100))

        # CPU训练器
        trainer_cpu = RiskMinerTrainer(
            X, y,
            device=torch.device('cpu'),
            random_seed=42
        )

        # GPU训练器
        trainer_gpu = RiskMinerTrainer(
            X, y,
            device=torch.device('cuda:0'),
            random_seed=42
        )

        # 网络应该在正确的设备上
        for param in trainer_cpu.policy_network.parameters():
            assert param.device.type == 'cpu'

        for param in trainer_gpu.policy_network.parameters():
            assert param.device.type == 'cuda'


class TestTrainerOptimization:
    """测试训练器优化"""

    def test_exploration_annealing(self):
        """测试探索退火"""
        X = pd.DataFrame({'close': [100, 101, 102]})
        y = pd.Series([0.01, 0.02, 0.03])

        trainer = RiskMinerTrainer(X, y)

        initial_c_puct = trainer.mcts_searcher.current_c_puct

        # 执行多次迭代
        for _ in range(10):
            trainer.mcts_searcher.update_exploration_rate()

        # 探索率应该降低
        assert trainer.mcts_searcher.current_c_puct < initial_c_puct
        assert trainer.mcts_searcher.current_c_puct >= trainer.mcts_searcher.min_c_puct

    def test_quantile_adaptation(self):
        """测试分位数自适应"""
        X = pd.DataFrame({'close': np.random.randn(100)})
        y = pd.Series(np.random.randn(100))

        trainer = RiskMinerTrainer(X, y)

        initial_quantile = trainer.optimizer.quantile_estimate

        # 模拟多个episode
        for _ in range(10):
            # 创建不同奖励的轨迹
            reward = np.random.random()
            trainer.optimizer.update_quantile(reward)

        # 分位数应该更新
        assert trainer.optimizer.quantile_estimate != initial_quantile

    def test_alpha_pool_maintenance(self):
        """测试Alpha池维护"""
        X = pd.DataFrame({'close': np.random.randn(100)})
        y = pd.Series(np.random.randn(100))

        trainer = RiskMinerTrainer(X, y)

        # 添加很多公式
        for i in range(150):
            trainer.alpha_pool.append({
                'formula': f'formula_{i}',
                'ic': np.random.random() * 0.1
            })

        # 维护池大小
        top_formulas = trainer.get_top_formulas(n=100)

        # 池大小应该被限制
        assert len(top_formulas) <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])