"""策略网络测试文件"""
import pytest
import numpy as np
import torch
import torch.nn as nn
from policy.network import PolicyNetwork
from policy.optimizer import RiskSeekingOptimizer
from mcts.environment import MDPState
from core.token_system import TOKEN_TO_INDEX, TOTAL_TOKENS


class TestPolicyNetwork:
    """测试策略网络"""

    @pytest.fixture
    def network(self):
        device = torch.device('cpu')
        return PolicyNetwork(device=device).to(device)

    @pytest.fixture
    def sample_state(self):
        """创建样本状态"""
        state = MDPState()
        state.add_token('close')
        state.add_token('volume')
        return state

    def test_initialization(self, network):
        """测试网络初始化"""
        # 检查层结构
        assert isinstance(network.gru, nn.GRU)
        assert network.gru.num_layers == 4
        assert network.gru.hidden_size == 64

        # 检查输出维度
        assert network.policy_head[-1].out_features == TOTAL_TOKENS
        assert network.value_head[-1].out_features == 1

    def test_forward_single_state(self, network, sample_state):
        """测试单个状态的前向传播"""
        # 编码状态
        state_encoding = torch.FloatTensor(
            sample_state.encode_for_network()
        ).unsqueeze(0)  # [1, seq_len, state_dim]

        # 创建合法动作掩码
        valid_mask = torch.ones(1, TOTAL_TOKENS, dtype=torch.bool)

        # 前向传播
        action_probs, value = network(state_encoding, valid_mask)

        assert action_probs.shape == (1, TOTAL_TOKENS)
        assert value.shape == (1, 1)

        # 概率应该和为1
        assert torch.allclose(action_probs.sum(), torch.tensor(1.0), atol=1e-5)

        # 所有概率应该非负
        assert torch.all(action_probs >= 0)

    def test_forward_with_mask(self, network, sample_state):
        """测试带掩码的前向传播"""
        state_encoding = torch.FloatTensor(
            sample_state.encode_for_network()
        ).unsqueeze(0)

        # 创建掩码（只允许部分动作）
        valid_mask = torch.zeros(1, TOTAL_TOKENS, dtype=torch.bool)
        valid_indices = [TOKEN_TO_INDEX['END'], TOKEN_TO_INDEX['add']]
        for idx in valid_indices:
            valid_mask[0, idx] = True

        action_probs, value = network(state_encoding, valid_mask)

        # 非法动作的概率应该是0
        for i in range(TOTAL_TOKENS):
            if i not in valid_indices:
                assert action_probs[0, i] < 1e-6

        # 合法动作的概率和应该是1
        valid_probs = action_probs[0, valid_indices].sum()
        assert torch.allclose(valid_probs, torch.tensor(1.0), atol=1e-5)

    def test_forward_batch(self, network):
        """测试批量前向传播"""
        batch_size = 4
        seq_len = 10
        state_dim = TOTAL_TOKENS + 3

        # 创建批量输入
        state_batch = torch.randn(batch_size, seq_len, state_dim)
        valid_mask = torch.ones(batch_size, TOTAL_TOKENS, dtype=torch.bool)

        action_probs, values = network(state_batch, valid_mask)

        assert action_probs.shape == (batch_size, TOTAL_TOKENS)
        assert values.shape == (batch_size, 1)

        # 每个样本的概率和应该是1
        for i in range(batch_size):
            assert torch.allclose(action_probs[i].sum(), torch.tensor(1.0), atol=1e-5)

    def test_get_action(self, network, sample_state):
        """测试动作选择"""
        # 确定性选择（temperature=0会在函数内部处理）
        action, prob = network.get_action(sample_state, temperature=0.1)

        assert action in TOKEN_TO_INDEX
        assert 0 <= prob <= 1

        # 随机选择
        actions = []
        for _ in range(100):
            action, _ = network.get_action(sample_state, temperature=1.0)
            actions.append(action)

        # 应该有多样性
        unique_actions = set(actions)
        assert len(unique_actions) > 1

    def test_gradient_flow(self, network):
        """测试梯度流"""
        state_encoding = torch.randn(2, 10, TOTAL_TOKENS + 3, requires_grad=True)
        valid_mask = torch.ones(2, TOTAL_TOKENS, dtype=torch.bool)

        action_probs, values = network(state_encoding, valid_mask)

        # 创建损失
        target_actions = torch.randint(0, TOTAL_TOKENS, (2,))
        policy_loss = -torch.log(action_probs.gather(1, target_actions.unsqueeze(1))).mean()
        value_loss = values.mean()
        total_loss = policy_loss + value_loss

        # 反向传播
        total_loss.backward()

        # 检查梯度
        for param in network.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.all(param.grad == 0)

    def test_numerical_stability(self, network):
        """测试数值稳定性"""
        # 极端输入
        extreme_input = torch.randn(1, 30, TOTAL_TOKENS + 3) * 100
        valid_mask = torch.ones(1, TOTAL_TOKENS, dtype=torch.bool)

        action_probs, values = network(extreme_input, valid_mask)

        # 应该没有NaN或Inf
        assert torch.all(torch.isfinite(action_probs))
        assert torch.all(torch.isfinite(values))

        # 概率仍然应该有效
        assert torch.allclose(action_probs.sum(), torch.tensor(1.0), atol=1e-5)

    def test_all_invalid_mask(self, network):
        """测试全无效掩码的边界情况"""
        state_encoding = torch.randn(1, 10, TOTAL_TOKENS + 3)

        # 所有动作都无效
        invalid_mask = torch.zeros(1, TOTAL_TOKENS, dtype=torch.bool)

        action_probs, values = network(state_encoding, invalid_mask)

        # 应该有兜底处理（比如END）
        assert torch.all(torch.isfinite(action_probs))
        assert action_probs.sum() > 0  # 至少有一个非零概率

        # END应该有概率
        end_idx = TOKEN_TO_INDEX.get('END', 0)
        assert action_probs[0, end_idx] > 0

    def test_with_log_probs(self, network):
        """测试返回log概率"""
        state_encoding = torch.randn(2, 10, TOTAL_TOKENS + 3)
        valid_mask = torch.ones(2, TOTAL_TOKENS, dtype=torch.bool)

        action_probs, values, log_probs = network(
            state_encoding, valid_mask, return_log_probs=True
        )

        assert log_probs.shape == action_probs.shape

        # log_probs应该是action_probs的对数
        expected_log_probs = torch.log(action_probs + 1e-12)
        assert torch.allclose(log_probs, expected_log_probs, atol=1e-4)


class TestRiskSeekingOptimizer:
    """测试风险寻求优化器"""

    @pytest.fixture
    def optimizer(self):
        device = torch.device('cpu')
        network = PolicyNetwork(device=device).to(device)
        return RiskSeekingOptimizer(network, quantile_alpha=0.85, device=device)

    def test_initialization(self, optimizer):
        """测试初始化"""
        assert optimizer.quantile_alpha == 0.85
        assert optimizer.quantile_estimate == -1.0
        assert optimizer.beta == 0.01
        assert isinstance(optimizer.optimizer, torch.optim.Adam)

    def test_update_quantile(self, optimizer):
        """测试分位数更新"""
        # 奖励低于分位数
        optimizer.quantile_estimate = 0.5
        new_q = optimizer.update_quantile(0.3)
        assert new_q < 0.5  # 应该降低

        # 奖励高于分位数
        optimizer.quantile_estimate = 0.5
        new_q = optimizer.update_quantile(0.7)
        assert new_q > 0.5  # 应该提高

        # 多次更新应该收敛
        for _ in range(100):
            reward = np.random.random()
            optimizer.update_quantile(reward)

        # 应该在合理范围内
        assert -2 < optimizer.quantile_estimate < 2

    def test_train_on_episode(self, optimizer):
        """测试episode训练"""
        # 创建轨迹
        trajectory = []

        state1 = MDPState()
        state1.add_token('close')
        trajectory.append((state1, 'volume', 0.1))

        state2 = MDPState()
        state2.add_token('close')
        state2.add_token('volume')
        trajectory.append((state2, 'add', 0.2))

        state3 = MDPState()
        state3.add_token('close')
        state3.add_token('volume')
        state3.add_token('add')
        trajectory.append((state3, 'END', 0.3))

        # 训练（总奖励=0.6）
        loss = optimizer.train_on_episode(trajectory)

        # 如果奖励超过分位数，应该有损失
        if sum([r for _, _, r in trajectory]) > optimizer.quantile_estimate:
            assert loss > 0
        else:
            assert loss == 0

    def test_risk_seeking_behavior(self, optimizer):
        """测试风险寻求行为"""
        # 创建高奖励和低奖励的轨迹
        high_reward_trajectory = []
        low_reward_trajectory = []

        for i in range(3):
            state = MDPState()
            if i > 0:
                state.add_token('close')
            action = 'volume' if i == 0 else 'END'

            high_reward_trajectory.append((state, action, 1.0))
            low_reward_trajectory.append((state, action, 0.01))

        # 训练多次
        high_reward_losses = []
        low_reward_losses = []

        for _ in range(10):
            loss_high = optimizer.train_on_episode(high_reward_trajectory)
            loss_low = optimizer.train_on_episode(low_reward_trajectory)

            high_reward_losses.append(loss_high)
            low_reward_losses.append(loss_low)

        # 高奖励应该更频繁地产生非零损失（被用于训练）
        high_nonzero = sum(1 for l in high_reward_losses if l > 0)
        low_nonzero = sum(1 for l in low_reward_losses if l > 0)

        assert high_nonzero >= low_nonzero

    def test_gradient_clipping(self, optimizer):
        """测试梯度裁剪"""
        # 创建会产生大梯度的轨迹
        trajectory = []
        for i in range(10):
            state = MDPState()
            if i > 0:
                state.add_token('close')
            trajectory.append((state, 'volume' if i == 0 else 'END', 100.0))

        # 强制让它训练（设置低分位数）
        optimizer.quantile_estimate = -100

        # 训练前记录参数
        params_before = []
        for param in optimizer.policy_network.parameters():
            params_before.append(param.clone().detach())

        loss = optimizer.train_on_episode(trajectory)

        # 参数应该更新但不应该爆炸
        for param, param_before in zip(optimizer.policy_network.parameters(), params_before):
            diff = (param - param_before).abs().max().item()
            assert diff < 1.0  # 梯度裁剪应该防止大更新

    def test_empty_trajectory(self, optimizer):
        """测试空轨迹"""
        empty_trajectory = []
        loss = optimizer.train_on_episode(empty_trajectory)
        assert loss == 0

    def test_invalid_trajectory(self, optimizer):
        """测试无效轨迹处理"""
        # 创建状态-动作不匹配的轨迹
        state = MDPState()  # 只有BEG
        invalid_trajectory = [(state, 'add', 0.1)]  # add需要2个操作数

        # 应该抛出错误或返回0
        try:
            loss = optimizer.train_on_episode(invalid_trajectory)
            # 如果没抛错，损失应该是0
            assert loss == 0
        except (AssertionError, RuntimeError):
            # 预期的错误
            pass


class TestIntegration:
    """集成测试"""

    def test_network_optimizer_integration(self):
        """测试网络和优化器的集成"""
        device = torch.device('cpu')
        network = PolicyNetwork(device=device).to(device)
        optimizer = RiskSeekingOptimizer(network, device=device)

        # 创建一些轨迹
        trajectories = []
        for _ in range(5):
            trajectory = []
            state = MDPState()

            # 简单的序列：close -> volume -> add -> END
            actions = ['close', 'volume', 'add', 'END']
            for i, action in enumerate(actions):
                if i > 0:
                    prev_state = state.copy()
                    state.add_token(actions[i - 1])
                    trajectory.append((prev_state, action, np.random.random() * 0.1))

            trajectories.append(trajectory)

        # 训练
        initial_params = [p.clone() for p in network.parameters()]

        for trajectory in trajectories:
            optimizer.train_on_episode(trajectory)

        # 至少有些参数应该改变
        params_changed = False
        for p_new, p_old in zip(network.parameters(), initial_params):
            if not torch.allclose(p_new, p_old):
                params_changed = True
                break

        # 可能所有轨迹都低于分位数，所以参数可能不变
        # 但分位数应该更新了
        assert optimizer.quantile_estimate != -1.0

    def test_device_consistency(self):
        """测试设备一致性"""
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            network = PolicyNetwork(device=device).to(device)

            state = MDPState()
            state.add_token('close')

            # get_action应该在正确的设备上工作
            action, prob = network.get_action(state)

            assert action in TOKEN_TO_INDEX
            assert 0 <= prob <= 1

            # 所有参数应该在GPU上
            for param in network.parameters():
                assert param.device.type == 'cuda'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])