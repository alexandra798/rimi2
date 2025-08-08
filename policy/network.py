"""Alpha挖掘策略网络"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

from core import TOKEN_TO_INDEX, INDEX_TO_TOKEN, TOTAL_TOKENS, RPNValidator


class PolicyNetwork(nn.Module):
    """策略网络：学习选择下一个Token（包括END）"""

    def __init__(self, state_dim=TOTAL_TOKENS + 3, action_dim=TOTAL_TOKENS, device=None):
        super().__init__()
        self.device = device if device else torch.device('cpu')

        # GRU编码器（处理Token序列）
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=64,
            num_layers=4,  # 论文指定4层
            batch_first=True,
            dropout=0.1
        )


        # 策略头（输出每个Token的选择概率）
        self.policy_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, action_dim)  # 输出所有Token的logits
        )

        # 价值头（可选，用于评估状态价值）
        self.value_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, state_encoding, valid_actions_mask=None):
        """
        前向传播

        Args:
            state_encoding: [batch_size, seq_len, state_dim] 状态编码
            valid_actions_mask: [batch_size, action_dim] 合法动作掩码

        Returns:
            action_probs: [batch_size, action_dim] 动作概率分布
            state_value: [batch_size, 1] 状态价值估计
        """
        # GRU编码
        gru_out, _ = self.gru(state_encoding)

        # 取最后一个时间步的输出
        last_hidden = gru_out[:, -1, :]  # [batch_size, 64]

        # 计算动作logits
        action_logits = self.policy_head(last_hidden)  # [batch_size, action_dim]

        # 应用合法动作掩码
        if valid_actions_mask is not None:
            # 将非法动作的logits设为-inf
            action_logits = action_logits.masked_fill(~valid_actions_mask, -1e9)

        # 转换为概率分布
        action_probs = F.softmax(action_logits, dim=-1)

        # 计算状态价值
        state_value = self.value_head(last_hidden)

        return action_probs, state_value

    def get_action(self, state, temperature=1.0):
        """
        根据当前状态选择动作

        Args:
            state: MDPState对象
            temperature: 温度参数，控制探索程度

        Returns:
            action: 选择的Token名称
            action_prob: 该动作的概率
        """
        # 编码状态
        state_encoding = torch.FloatTensor(state.encode_for_network()).unsqueeze(0).to(self.device)

        # 获取合法动作
        valid_tokens = RPNValidator.get_valid_next_tokens(state.token_sequence)
        valid_actions_mask = torch.zeros(TOTAL_TOKENS, dtype=torch.bool)
        for token_name in valid_tokens:
            valid_actions_mask[TOKEN_TO_INDEX[token_name]] = True
        valid_actions_mask = valid_actions_mask.unsqueeze(0).to(self.device)

        # 前向传播
        with torch.no_grad():
            action_probs, _ = self.forward(state_encoding, valid_actions_mask)

        # 温度缩放
        if temperature != 1.0:
            action_probs = F.softmax(torch.log(action_probs + 1e-10) / temperature, dim=-1)

        # 采样动作
        action_idx = torch.multinomial(action_probs[0], 1).item()
        action_name = INDEX_TO_TOKEN[action_idx]
        action_prob = action_probs[0, action_idx].item()

        return action_name, action_prob