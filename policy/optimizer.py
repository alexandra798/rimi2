"""风险寻求策略优化器"""
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

from core import TOKEN_TO_INDEX, RPNValidator


class RiskSeekingOptimizer:
    """风险寻求策略优化（优化最好情况而非平均情况）"""

    def __init__(self, policy_network, quantile_alpha=0.85, device=None):
        self.policy_network = policy_network
        self.quantile_alpha = quantile_alpha  # 目标分位数（如0.85表示优化top 15%）
        self.quantile_estimate = -1.0  # 当前分位数估计
        self.beta = 0.01  # 分位数更新学习率
        self.device = device if device else torch.device('cpu')

        self.optimizer = torch.optim.Adam(
            policy_network.parameters(),
            lr=0.001  # 网络参数学习率
        )

    def update_quantile(self, episode_reward):
        """
        更新分位数估计
        公式：q_{i+1} = q_i + β(1 - α - 1{R(τ_i) ≤ q_i})
        """
        indicator = 1.0 if episode_reward <= self.quantile_estimate else 0.0
        self.quantile_estimate += self.beta * (1 - self.quantile_alpha - indicator)
        return self.quantile_estimate

    def train_on_episode(self, episode_trajectory):
        """
        使用一个episode的轨迹训练策略网络

        Args:
            episode_trajectory: [(state, action, reward), ...] 列表
        """
        # 计算episode总奖励
        total_reward = sum([r for _, _, r in episode_trajectory])

        # 更新分位数估计
        self.update_quantile(total_reward)

        # 只有当奖励超过分位数时才更新（风险寻求）
        if total_reward > self.quantile_estimate:
            # 构建训练数据
            states = []
            actions = []
            rewards = []

            for state, action, reward in episode_trajectory:
                states.append(state.encode_for_network())
                actions.append(TOKEN_TO_INDEX[action])
                rewards.append(reward)

            # 转换为张量
            states_array = np.array(states)  # 先转换为单个numpy数组
            states_tensor = torch.FloatTensor(states_array).to(self.device)
            actions_tensor = torch.LongTensor(actions).to(self.device)

            # 计算累积奖励（可以使用折扣因子）
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + 0.99 * G  # 折扣因子0.99
                returns.insert(0, G)
            returns_tensor = torch.FloatTensor(returns).to(self.device)

            masks = []
            lengths = []
            for state, action, reward in episode_trajectory:
                valid = [False] * len(TOKEN_TO_INDEX)
                for name in RPNValidator.get_valid_next_tokens(state.token_sequence):
                    valid[TOKEN_TO_INDEX[name]] = True
                masks.append(valid)
                # 真实长度：token_sequence 长度（不超过 encode 的最大步长）
                lengths.append(min(len(state.token_sequence), states_tensor.size(1)))

            masks_tensor = torch.BoolTensor(masks).to(self.device)
            lengths_tensor = torch.LongTensor(lengths).to(self.device)

            # 前向传播
            action_probs, values = self.policy_network(states_tensor, valid_actions_mask=masks_tensor,
                                                       lengths=lengths_tensor)

            # 计算策略损失（REINFORCE with baseline）
            log_probs = torch.log(action_probs.gather(1, actions_tensor.unsqueeze(1)))
            advantages = returns_tensor - values.squeeze()
            policy_loss = -(log_probs.squeeze() * advantages.detach()).mean()

            # 修改终点 计算价值损失
            value_loss = F.mse_loss(values.squeeze(), returns_tensor)

            # 总损失
            total_loss = policy_loss + 0.5 * value_loss

            # 反向传播和优化
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
            self.optimizer.step()

            return total_loss.item()

        return 0.0