"""基于Token的MCTS搜索器"""
import numpy as np
import math
import logging
import torch
from core import RPNValidator, TOKEN_TO_INDEX, TOKEN_DEFINITIONS, TokenType


logger = logging.getLogger(__name__)


class MCTSSearcher:
    """MCTS搜索器 - 实现PUCT选择和树搜索"""

    def __init__(self, policy_network=None, device=None,
                 initial_c_puct=1.414, c_puct_decay=0.995, min_c_puct=0.5):
        self.policy_network = policy_network
        self.device = device
        self.gamma = 1.0

        # 探索退火参数
        self.initial_c_puct = initial_c_puct
        self.current_c_puct = initial_c_puct
        self.c_puct_decay = c_puct_decay
        self.min_c_puct = min_c_puct
        self.iteration_count = 0

    def update_exploration_rate(self):
        """更新探索率（退火）"""
        self.current_c_puct = max(
            self.min_c_puct,
            self.initial_c_puct * (self.c_puct_decay ** self.iteration_count)
        )
        self.iteration_count += 1

    def search_one_iteration(self, root_node, mdp_env, reward_calculator, X_data, y_data):
        #执行一次完整的MCTS迭代
        #四个阶段：Selection, Expansion, Rollout, Backpropagation
        import time


        # 阶段1：选择（Selection）
        start = time.time()
        logger.debug("Starting selection phase...")
        path = []
        current = root_node

        # 使用PUCT选择直到叶节点
        while current.is_expanded() and not current.is_terminal():
            current = current.get_best_child(c_puct=self.current_c_puct)
            if current is None:
                break
            path.append(current)

            # 如果当前序列合法，更新边的中间奖励R(s,a)
            if RPNValidator.is_valid_partial_expression(current.state.token_sequence):
                logger.debug(f"Calculating intermediate reward for path length {len(path)}...")
                reward_start = time.time()
                intermediate_reward = reward_calculator.calculate_intermediate_reward(
                    current.state, X_data, y_data
                )
                logger.debug(f"Intermediate reward calculation took {time.time() - reward_start:.2f}s")
                current.update_intermediate_reward(intermediate_reward)
        logger.debug(f"Selection phase took {time.time() - start:.2f}s")
        # 阶段2：扩展（Expansion）
        leaf_value = 0
        if not current.is_terminal() and current.N >= 0:
            # 扩展叶节点
            leaf_value = self.expand(current, mdp_env)

            # 选择一个新扩展的子节点进行评估
            if current.children:
                # 根据先验概率选择

                probs = [child.P for child in current.children.values()]
                probs = np.array(probs, dtype=np.float64)
                probs = np.where(np.isfinite(probs) & (probs >= 0.0), probs, 0.0)
                s = probs.sum()
                if (not np.isfinite(s)) or s <= 0.0:
                    probs = np.full(len(probs), 1.0 / len(probs))
                else:
                    probs = probs / s

                selected_idx = np.random.choice(len(current.children), p=probs)

                selected_action = list(current.children.keys())[selected_idx]
                current = current.children[selected_action]
                path.append(current)


        # 阶段3：Rollout
        if current.is_terminal():
            # 终止状态，计算终止奖励
            value = reward_calculator.calculate_terminal_reward(
                current.state, X_data, y_data
            )
        else:
            # 执行rollout评估叶节点价值
            value = self.rollout(current, mdp_env, reward_calculator, X_data, y_data)

        # 阶段4：回传（Backpropagation）- 按照论文公式
        self.backpropagate(path, value, reward_calculator, X_data, y_data)

        # 构建轨迹用于策略网络训练
        trajectory = self.extract_trajectory(path)

        self.update_exploration_rate()

        return trajectory

    def expand(self, node, mdp_env):
        """
        扩展节点，调用风险策略网络获取先验概率P(s,a)
        """
        if node.state is None:
            return 0

        # 获取所有合法动作
        valid_actions = mdp_env.get_valid_actions(node.state)

        if not valid_actions:
            return 0

        # 使用策略网络获取先验概率和价值估计
        if self.policy_network:
            action_probs, value_estimate = self.get_policy_predictions(node.state, valid_actions)
        else:
            # 均匀分布
            action_probs = {action: 1.0 / len(valid_actions) for action in valid_actions}
            value_estimate = 0

        # 为每个合法动作创建子节点，设置初始值
        for action in valid_actions:
            # 创建新状态
            new_state = node.state.copy()
            new_state.add_token(action)

            # 添加子节点，设置先验概率P(s,a)
            prior_prob = action_probs.get(action, 1.0 / len(valid_actions))
            node.add_child(action, new_state, prior_prob)

        return value_estimate


    def rollout(self, node, mdp_env, reward_calculator, X_data, y_data, max_depth=30):
        """
        执行rollout 使用策略网络作为rollout policy
        """
        current_state = node.state.copy()
        cumulative_reward = 0
        depth = 0
        intermediate_rewards = []

        while depth < max_depth and not current_state.token_sequence[-1].name == 'END':
            # 获取合法动作
            valid_actions = mdp_env.get_valid_actions(current_state)

            if not valid_actions:
                break

            # 使用策略网络选择动作（如果有）
            if self.policy_network:
                action_probs, _ = self.get_policy_predictions(current_state, valid_actions)

                probs = [action_probs.get(a, 0.0) for a in valid_actions]
                probs = np.array(probs, dtype=np.float64)
                probs = np.where(np.isfinite(probs) & (probs >= 0.0), probs, 0.0)
                s = probs.sum()
                if (not np.isfinite(s)) or s <= 0.0:
                    probs = np.full(len(valid_actions), 1.0 / len(valid_actions))
                else:
                    probs = probs / s
                action = np.random.choice(valid_actions, p=probs)

            else:
                # 随机选择
                action = np.random.choice(valid_actions)

            # 应用动作
            current_state.add_token(action)

            # 计算奖励
            if action == 'END':
                # 终止奖励
                reward = reward_calculator.calculate_terminal_reward(
                    current_state, X_data, y_data
                )
            else:
                # 中间奖励
                if RPNValidator.is_valid_partial_expression(current_state.token_sequence):
                    reward = reward_calculator.calculate_intermediate_reward(
                        current_state, X_data, y_data
                    )
                else:
                    reward = 0

            intermediate_rewards.append(reward)
            depth += 1

            if action == 'END':
                break

        # 计算累积奖励（论文：γ=1）
        v_l = 0  # 叶节点价值
        for reward in reversed(intermediate_rewards):
            v_l = reward + self.gamma * v_l

        return v_l

    #不保留智能rollout了


    def backpropagate(self, path, leaf_value, reward_calculator, X_data, y_data):
        """
        正确的Bootstrap回传实现
        G_k = Σ_{i=0}^{l-1-k} γ^i * r_{k+1+i} + γ^{l-k} * v_l
        """
        if not path:
            return

            # 收集路径上的奖励
        rewards = []
        for i, node in enumerate(path):
            if i > 0:  # 跳过根节点
                # 使用节点存储的中间奖励
                rewards.append(node.R)

        # l是最后一个节点的索引
        l = len(path) - 1

        # 对路径上的每个节点进行更新
        for k in range(len(path)):
            # 计算G_k：k位置的累积回报
            G_k = 0

            # 累加从k到l-1的折扣奖励
            for i in range(min(l - k, len(rewards) - k)):
                if k + i < len(rewards):
                    G_k += (self.gamma ** i) * rewards[k + i]

            # 加上叶节点的折扣值
            if l >= k:
                G_k += (self.gamma ** (l - k)) * leaf_value

            # 更新节点的Q值
            path[k].update(G_k)

    def extract_trajectory(self, path):
        """
        从路径中提取轨迹τ = {s_0, a_1, r_1, s_1, ..., s_T}
        用于策略网络训练
        """
        trajectory = []

        for i, node in enumerate(path):
            if node.parent and node.action:
                # (state, action, reward)
                trajectory.append((
                    node.parent.state,
                    node.action,
                    node.R  # 使用存储的中间奖励
                ))

        return trajectory

    # 此功能要使用transformer吗？思考这个问题
    def get_policy_predictions(self, state, valid_actions):
        """
        使用策略网络获取动作概率分布P(s,a)和价值估计
        """
        if not self.policy_network:
            probs = {action: 1.0 / len(valid_actions) for action in valid_actions}
            return probs, 0

        # 编码状态
        state_encoding = torch.FloatTensor(state.encode_for_network()).unsqueeze(0).to(self.device)

        # 创建合法动作掩码
        valid_actions_mask = torch.zeros(len(TOKEN_TO_INDEX), dtype=torch.bool)
        for action in valid_actions:
            valid_actions_mask[TOKEN_TO_INDEX[action]] = True
        valid_actions_mask = valid_actions_mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 不需要log_probs，只要action_probs
            result = self.policy_network(state_encoding, valid_actions_mask, return_log_probs=False)
            if len(result) == 3:
                action_probs, value, _ = result
            else:
                action_probs, value = result

        # 转换并清洗为概率字典（非负、有限、和为1；否则均匀分布）
        probs = {}
        raw = []
        for action in valid_actions:
            idx = TOKEN_TO_INDEX[action]
            p = float(action_probs[0, idx].item())
            if not (np.isfinite(p) and p >= 0.0):
                p = 0.0
            raw.append(p)

        raw = np.array(raw, dtype=np.float64)
        s = raw.sum()
        if (not np.isfinite(s)) or s <= 0.0:
            raw = np.full_like(raw, 1.0 / len(raw))
        else:
            raw = raw / s

        for action, p in zip(valid_actions, raw):
            probs[action] = float(p)

        return probs, float(value.item())

    def get_best_action(self, root_node, temperature=1.0):
        """
        根据访问次数选择最佳动作

        Args:
            root_node: 根节点
            temperature: 温度参数，控制选择的随机性

        Returns:
            action: 选择的动作
        """
        if not root_node.children:
            return None

        actions, visits = root_node.get_visit_distribution()

        if temperature == 0:
            # 贪婪选择
            best_idx = np.argmax(visits)
            return actions[best_idx]
        else:
            visits = np.array(visits, dtype=np.float64)
            if temperature != 1.0:
                visits = visits ** (1.0 / temperature)
            visits = np.where(np.isfinite(visits) & (visits >= 0.0), visits, 0.0)
            s = visits.sum()
            if (not np.isfinite(s)) or s <= 0.0:
                probs = np.full(len(actions), 1.0 / len(actions))
            else:
                probs = visits / s
            return np.random.choice(actions, p=probs)






