"""MCTS搜索测试文件"""
import pytest
import numpy as np
import pandas as pd
import torch
from mcts.node import MCTSNode
from mcts.searcher import MCTSSearcher
from mcts.environment import AlphaMiningMDP, MDPState
from mcts.reward_calculator import RewardCalculator
from core.token_system import TOKEN_DEFINITIONS


class TestMDPState:
    """测试MDP状态"""

    def test_initialization(self):
        """测试初始化"""
        state = MDPState()
        assert len(state.token_sequence) == 1
        assert state.token_sequence[0].name == 'BEG'
        assert state.step_count == 0
        assert state.stack_size == 0

    def test_add_token(self):
        """测试添加Token"""
        state = MDPState()

        # 添加操作数
        state.add_token('close')
        assert len(state.token_sequence) == 2
        assert state.token_sequence[1].name == 'close'
        assert state.step_count == 1
        assert state.stack_size == 1

        # 添加另一个操作数
        state.add_token('volume')
        assert state.stack_size == 2

        # 添加操作符
        state.add_token('add')
        assert state.stack_size == 1  # 2 - 2 + 1 = 1

    def test_add_invalid_beg(self):
        """测试重复添加BEG"""
        state = MDPState()
        state.add_token('close')

        with pytest.raises(ValueError):
            state.add_token('BEG')

    def test_encode_for_network(self):
        """测试状态编码"""
        state = MDPState()
        state.add_token('close')
        state.add_token('volume')
        state.add_token('add')

        encoding = state.encode_for_network()

        assert encoding.shape == (30, len(TOKEN_DEFINITIONS) + 3)
        # 第一个位置应该编码BEG
        assert encoding[0, TOKEN_DEFINITIONS.__len__() - 1] <= 1  # 有BEG的one-hot编码
        # 后面应该有其他信息
        assert encoding[1, TOKEN_DEFINITIONS.__len__()] > 0  # 位置信息

    def test_copy(self):
        """测试深拷贝"""
        state = MDPState()
        state.add_token('close')
        state.add_token('volume')

        state_copy = state.copy()

        # 修改副本不应影响原始
        state_copy.add_token('add')

        assert len(state.token_sequence) == 2
        assert len(state_copy.token_sequence) == 3
        assert state.stack_size == 2
        assert state_copy.stack_size == 1


class TestAlphaMiningMDP:
    """测试MDP环境"""

    @pytest.fixture
    def mdp(self):
        return AlphaMiningMDP()

    def test_reset(self, mdp):
        """测试重置"""
        state = mdp.reset()
        assert isinstance(state, MDPState)
        assert len(state.token_sequence) == 1
        assert state.token_sequence[0].name == 'BEG'

    def test_step(self, mdp):
        """测试执行动作"""
        mdp.reset()

        # 执行有效动作
        state, reward, done = mdp.step('close')
        assert state is not None
        assert done == False

        # 执行END
        state, reward, done = mdp.step('END')
        assert done == True

    def test_invalid_action(self, mdp):
        """测试无效动作"""
        mdp.reset()

        # 尝试无效动作（需要操作数）
        state, reward, done = mdp.step('add')
        assert reward == -1.0
        assert done == True

    def test_is_valid_action(self, mdp):
        """测试动作有效性检查"""
        state = mdp.reset()

        assert mdp.is_valid_action('close') == True
        assert mdp.is_valid_action('volume') == True
        assert mdp.is_valid_action('add') == False  # 需要2个操作数
        assert mdp.is_valid_action('END') == False  # 不能直接结束

    def test_get_valid_actions(self, mdp):
        """测试获取有效动作"""
        state = mdp.reset()

        valid_actions = mdp.get_valid_actions(state)
        assert 'close' in valid_actions
        assert 'volume' in valid_actions
        assert 'END' not in valid_actions

        # 添加一个操作数后
        state.add_token('close')
        valid_actions = mdp.get_valid_actions(state)
        assert 'END' in valid_actions  # 现在可以结束了

    def test_max_episode_length(self, mdp):
        """测试最大episode长度"""
        state = mdp.reset()

        # 添加接近最大长度的Token
        for i in range(28):
            state, reward, done = mdp.step('close' if i % 2 == 0 else 'volume')
            if i < 27:
                assert done == False

        # 达到最大长度
        state, reward, done = mdp.step('add')
        assert done == True  # 应该强制结束

    def test_constant_filtering(self, mdp):
        """测试常数过滤"""
        state = mdp.reset()
        state.add_token('const_1')  # 添加常数

        valid_actions = mdp.get_valid_actions(state)

        # 某些操作符不应该对常数可用
        assert 'ts_std' not in valid_actions  # 常数的标准差还是常数
        assert 'ts_var' not in valid_actions

        # 但基本运算应该可用
        assert 'add' in valid_actions or 'close' in valid_actions


class TestMCTSNode:
    """测试MCTS节点"""

    def test_initialization(self):
        """测试初始化"""
        state = MDPState()
        node = MCTSNode(state=state)

        assert node.state == state
        assert node.parent is None
        assert node.N == 0
        assert node.Q == 0.0
        assert len(node.children) == 0

    def test_add_child(self):
        """测试添加子节点"""
        parent_state = MDPState()
        parent = MCTSNode(state=parent_state)

        child_state = MDPState()
        child_state.add_token('close')

        child = parent.add_child('close', child_state, prior_prob=0.5)

        assert 'close' in parent.children
        assert parent.children['close'] == child
        assert child.parent == parent
        assert child.action == 'close'
        assert child.P == 0.5

    def test_update(self):
        """测试更新节点"""
        node = MCTSNode(state=MDPState())

        # 第一次更新
        node.update(1.0)
        assert node.N == 1
        assert node.Q == 1.0

        # 第二次更新
        node.update(0.5)
        assert node.N == 2
        assert node.Q == 0.75  # (1.0 + 0.5) / 2

    def test_get_best_child(self):
        """测试选择最佳子节点"""
        parent = MCTSNode(state=MDPState())

        # 添加多个子节点
        states = [MDPState() for _ in range(3)]
        for i, (action, state) in enumerate(zip(['close', 'volume', 'open'], states)):
            state.add_token(action)
            child = parent.add_child(action, state, prior_prob=0.3)
            # 给不同的访问次数和Q值
            for _ in range(i + 1):
                child.update(0.1 * (i + 1))

        # 选择最佳子节点
        best = parent.get_best_child(c_puct=1.0)
        assert best is not None

        # 初次选择时应该考虑先验概率
        parent_new = MCTSNode(state=MDPState())
        for action in ['close', 'volume']:
            state = MDPState()
            state.add_token(action)
            parent_new.add_child(action, state, prior_prob=0.8 if action == 'close' else 0.2)

        best = parent_new.get_best_child()
        assert best.action == 'close'  # 应该选择先验概率高的

    def test_is_terminal(self):
        """测试终止检查"""
        state = MDPState()
        node = MCTSNode(state=state)
        assert node.is_terminal() == False

        state.add_token('close')
        state.add_token('END')
        node = MCTSNode(state=state)
        assert node.is_terminal() == True

    def test_visit_distribution(self):
        """测试访问分布"""
        parent = MCTSNode(state=MDPState())

        # 添加子节点并更新访问次数
        for action in ['close', 'volume', 'open']:
            state = MDPState()
            state.add_token(action)
            child = parent.add_child(action, state)

            # 不同的访问次数
            visits = {'close': 10, 'volume': 5, 'open': 2}
            for _ in range(visits[action]):
                child.update(0.1)

        actions, visits = parent.get_visit_distribution()

        assert len(actions) == 3
        assert len(visits) == 3
        # 访问次数应该匹配
        visit_dict = dict(zip(actions, visits))
        assert visit_dict['close'] == 10
        assert visit_dict['volume'] == 5
        assert visit_dict['open'] == 2


class TestRewardCalculator:
    """测试奖励计算器"""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({
            'close': np.random.randn(n) * 10 + 100,
            'volume': np.random.exponential(1000000, n)
        })
        y = pd.Series(np.random.randn(n) * 0.01)
        return X, y

    @pytest.fixture
    def reward_calculator(self):
        alpha_pool = []
        return RewardCalculator(
            alpha_pool,
            lambda_param=0.1,
            sample_size=50,
            min_std=1e-6
        )

    def test_is_nearly_constant(self, reward_calculator):
        """测试常数检测"""
        # 常数
        constant = pd.Series([1.0] * 100)
        assert reward_calculator.is_nearly_constant(constant) == True

        # 非常数
        varying = pd.Series(np.random.randn(100))
        assert reward_calculator.is_nearly_constant(varying) == False

        # 极小变化
        tiny_var = pd.Series([1.0] * 99 + [1.0 + 1e-10])
        assert reward_calculator.is_nearly_constant(tiny_var) == True

    def test_calculate_ic(self, reward_calculator):
        """测试IC计算"""
        # 正相关
        pred = pd.Series([1, 2, 3, 4, 5])
        target = pd.Series([2, 4, 6, 8, 10])
        ic = reward_calculator.calculate_ic(pred, target)
        assert abs(ic - 1.0) < 0.01

        # 处理NaN
        pred_nan = pd.Series([1, np.nan, 3, 4, 5])
        target_nan = pd.Series([2, 4, np.nan, 8, 10])
        ic = reward_calculator.calculate_ic(pred_nan, target_nan)
        assert np.isfinite(ic)

        # 常数序列
        constant = pd.Series([1.0] * 100)
        varying = pd.Series(np.random.randn(100))
        ic = reward_calculator.calculate_ic(constant, varying)
        assert ic == 0.0

    def test_intermediate_reward(self, reward_calculator, sample_data):
        """测试中间奖励计算"""
        X, y = sample_data

        # 创建有效状态
        state = MDPState()
        state.add_token('close')

        reward = reward_calculator.calculate_intermediate_reward(state, X, y)

        assert isinstance(reward, float)
        assert np.isfinite(reward)

        # 无效状态应该返回负奖励
        invalid_state = MDPState()
        invalid_state.add_token('add')  # 无效：add需要2个操作数

        reward = reward_calculator.calculate_intermediate_reward(invalid_state, X, y)
        assert reward < 0

    def test_terminal_reward(self, reward_calculator, sample_data):
        """测试终止奖励"""
        X, y = sample_data

        # 创建终止状态
        state = MDPState()
        state.add_token('close')
        state.add_token('END')

        reward = reward_calculator.calculate_terminal_reward(state, X, y)

        assert isinstance(reward, float)
        assert np.isfinite(reward)

        # 非终止状态应该返回负奖励
        non_terminal = MDPState()
        non_terminal.add_token('close')

        reward = reward_calculator.calculate_terminal_reward(non_terminal, X, y)
        assert reward < 0

    def test_reward_cache(self, reward_calculator, sample_data):
        """测试奖励缓存"""
        X, y = sample_data

        state = MDPState()
        state.add_token('close')

        # 第一次计算
        reward1 = reward_calculator.calculate_intermediate_reward(state, X, y)

        # 第二次计算（应该使用缓存）
        reward2 = reward_calculator.calculate_intermediate_reward(state, X, y)

        assert reward1 == reward2
        assert len(reward_calculator._cache) > 0


class TestMCTSSearcher:
    """测试MCTS搜索器"""

    @pytest.fixture
    def searcher(self):
        device = torch.device('cpu')
        return MCTSSearcher(device=device)

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({
            'close': np.random.randn(n) * 10 + 100,
            'volume': np.random.exponential(1000000, n)
        })
        y = pd.Series(np.random.randn(n) * 0.01)
        return X, y

    def test_initialization(self, searcher):
        """测试初始化"""
        assert searcher.gamma == 1.0
        assert searcher.current_c_puct > 0
        assert searcher.iteration_count == 0

    def test_update_exploration_rate(self, searcher):
        """测试探索率更新"""
        initial_c = searcher.current_c_puct

        searcher.update_exploration_rate()
        assert searcher.current_c_puct <= initial_c
        assert searcher.iteration_count == 1

        # 多次更新后应该接近最小值
        for _ in range(100):
            searcher.update_exploration_rate()

        assert searcher.current_c_puct >= searcher.min_c_puct
        assert searcher.current_c_puct <= searcher.initial_c_puct

    def test_expand(self, searcher):
        """测试节点扩展"""
        mdp = AlphaMiningMDP()
        state = mdp.reset()
        node = MCTSNode(state=state)

        value = searcher.expand(node, mdp)

        assert len(node.children) > 0
        # 所有子节点应该有先验概率
        for child in node.children.values():
            assert child.P > 0
            assert child.P <= 1

    def test_get_best_action(self, searcher):
        """测试选择最佳动作"""
        root = MCTSNode(state=MDPState())

        # 添加子节点并设置不同的访问次数
        for action, visits in [('close', 10), ('volume', 5), ('open', 2)]:
            state = MDPState()
            state.add_token(action)
            child = root.add_child(action, state)
            child.N = visits

        # 贪婪选择（temperature=0）
        action = searcher.get_best_action(root, temperature=0)
        assert action == 'close'  # 访问次数最多

        # 随机选择（temperature>0）
        actions = []
        for _ in range(100):
            action = searcher.get_best_action(root, temperature=1.0)
            actions.append(action)

        # 应该有多样性
        assert len(set(actions)) > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])