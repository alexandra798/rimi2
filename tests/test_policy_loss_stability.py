import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, numpy as np
from policy.network import PolicyNetwork
from policy.optimizer import RiskSeekingOptimizer
from mcts.environment import MDPState

net = PolicyNetwork()
opt = RiskSeekingOptimizer(net)

# 伪造一条episode，动作合法且不会出现 log(0)
# 前状态：BEG, close, open, add
state = MDPState()
state.add_token('close')
state.add_token('open')
state.add_token('add')      # 二元把两个操作数归约为1
pre_state = state.copy()

episode = [(pre_state, 'END', 1.0)]

loss = opt.train_on_episode(episode)
print("loss:", loss)

# UserWarning: Using a target size (torch.Size([1])) that is different to the input size (torch.Size([])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
#   value_loss = F.mse_loss(values.squeeze(), returns_tensor)
