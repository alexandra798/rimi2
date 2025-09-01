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
        self.quantile_alpha = quantile_alpha  # 目标分位数
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
        Args: episode_trajectory: [(state, action, reward), ...] 列表
        """

        if gamma is None:
            gamma = self.gamma

        # 计算episode总奖励
        total_reward = sum([r for _, _, r in episode_trajectory])

        # 更新分位数估计
        self.update_quantile(total_reward)
        updated = False
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
                    pre_state.token_sequence.pop()
                    pre_state.step_count = max(0, pre_state.step_count - 1)
                    pre_state.stack_size = RPNValidator.calculate_stack_size(pre_state.token_sequence)

                # 取合法动作集并做强校验
                valid_tokens = RPNValidator.get_valid_next_tokens(pre_state.token_sequence)
                if action not in valid_tokens:
                    import logging
                    logging.debug(f"Skip illegal pair in training: action={action}, valid={valid_tokens}")
                    continue

                # 编码 & 收集
                states_enc.append(pre_state.encode_for_network())
                actions_idx.append(TOKEN_TO_INDEX[action])

                row_mask = [False] * len(TOKEN_TO_INDEX)
                for name in valid_tokens:
                    row_mask[TOKEN_TO_INDEX[name]] = True
                masks.append(row_mask)

                # 实际长度=token_sequence长度（不超过编码最大步长30）
                lengths.append(min(len(pre_state.token_sequence), 30))

            if not states_enc:
                return 0.0

            # 转换为张量
            states_tensor = torch.as_tensor(np.array(states_enc), dtype=torch.float32, device=self.device)
            actions_tensor = torch.as_tensor(actions_idx, dtype=torch.long, device=self.device)
            masks_tensor = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
            lengths_tensor = torch.as_tensor(lengths, dtype=torch.long, device=self.device)

            action_probs, log_probs = self.policy_network(
                states_tensor,
                valid_actions_mask=masks_tensor,
                lengths=lengths_tensor,
                return_log_probs=True
            )
            # 取每个动作的log概率
            chosen_log_probs = log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

            # 验证数值稳定性
            if not torch.isfinite(chosen_log_probs).all():
                bad = (~torch.isfinite(chosen_log_probs)).nonzero(as_tuple=False).squeeze(-1).tolist()
                raise RuntimeError(f"Chosen log-prob is not finite at indices {bad}")

            loss = -chosen_log_probs.sum()   # 注意是sum不是mean（按照论文公式）

            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)

            self.optimizer.step()
            updated = True
            return updated, float(loss.item())


        return updated,0.0