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
        self.gamma = 1.0

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

    def train_on_episode(self, episode_trajectory, gamma=None):
        """
        使用一个episode的轨迹训练策略网络
        Args: episode_trajectory: [(state, action, reward), ...] 列表
        """
        if gamma is None:
            gamma = self.gamma
        # 计算episode总奖励
        total_reward = sum([r for _, _, r in episode_trajectory])
        # 更新分位数估计
        self.update_quantile(total_reward)

        # 只有当奖励超过分位数时才更新（风险寻求）
        if total_reward > self.quantile_estimate:

            # 构建训练数据
            states_enc = []
            actions_idx = []
            rewards = []
            masks = []
            lengths = []

            for state, action, reward in episode_trajectory:
                pre_state = state
                # 若发现轨迹里存的是“后状态”，尝试自动回退一步（常见于把 END 先写进 state 再当作动作）
                if len(pre_state.token_sequence) >= 2 and pre_state.token_sequence[-1].name == action:
                    pre_state = pre_state.copy()
                    # 回退最后一个 token
                    pre_state.token_sequence.pop()
                    pre_state.step_count = max(0, pre_state.step_count - 1)
                    # 依据 token 序列重算栈高
                    pre_state.stack_size = RPNValidator.calculate_stack_size(pre_state.token_sequence)
                # 取合法动作集并做强校验
                valid_tokens = RPNValidator.get_valid_next_tokens(pre_state.token_sequence)
                if action not in valid_tokens:
                    # 记录一下，跳过该条，不要让整次训练崩
                    import logging
                    logging.debug(f"Skip illegal pair in training: action={action}, valid={valid_tokens}")
                    continue

                # 编码 & 收集
                states_enc.append(pre_state.encode_for_network())
                actions_idx.append(TOKEN_TO_INDEX[action])
                rewards.append(reward)

                row_mask = [False] * len(TOKEN_TO_INDEX)
                for name in valid_tokens:
                    row_mask[TOKEN_TO_INDEX[name]] = True
                masks.append(row_mask)

                # 实际长度=token_sequence长度（不超过编码最大步长30）
                lengths.append(min(len(pre_state.token_sequence), 30))

            # 转换为张量
            states_tensor = torch.as_tensor(np.array(states_enc), dtype=torch.float32, device=self.device)
            actions_tensor = torch.as_tensor(actions_idx, dtype=torch.long, device=self.device)
            masks_tensor = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
            lengths_tensor = torch.as_tensor(lengths, dtype=torch.long, device=self.device)

            # 计算累积奖励（可以使用折扣因子）
            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

            masks_tensor = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
            lengths_tensor = torch.as_tensor(lengths, dtype=torch.long, device=self.device)

            # 前向传播
            action_probs, values, log_probs = self.policy_network(
                states_tensor,
                valid_actions_mask=masks_tensor,
                lengths=lengths_tensor,
                return_log_probs=True
            )

            # 计算策略损失（REINFORCE with baseline）
            # 取每条轨迹动作的 log-prob
            chosen_log_probs = log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

            # 如果仍然出现 -inf，立刻抛错（一定是状态-动作不匹配没修干净）
            if not torch.isfinite(chosen_log_probs).all():
                bad = (~torch.isfinite(chosen_log_probs)).nonzero(as_tuple=False).squeeze(-1).tolist()
                raise RuntimeError(f"Chosen log-prob is not finite at indices {bad}; "
                                   f"check episode states are pre-action and action valid.")

            values_flat = values.squeeze(-1)
            assert values_flat.shape == returns_tensor.shape, \
                f"value/return shape mismatch: {values_flat.shape} vs {returns_tensor.shape}"
            advantages = returns_tensor - values_flat
            policy_loss = -(chosen_log_probs * advantages.detach()).mean()

            # 修改终点 计算价值损失

            value_loss = F.mse_loss(values_flat, returns_tensor)
            # UserWarning: Using a target size (torch.Size([1])) that is different to the input size (torch.Size([])).
            # This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
            # value_loss = F.mse_loss(values.squeeze(), returns_tensor)

            # 总损失
            total_loss = policy_loss + 0.5 * value_loss

            # 反向传播和优化
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
            self.optimizer.step()

            return total_loss.item()

        return 0.0