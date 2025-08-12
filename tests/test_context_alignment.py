import pandas as pd, numpy as np
from alpha.evaluator import FormulaEvaluator
from mcts.reward_calculator import RewardCalculator

# 构造两段不同长度的上下文
N1, N2 = 1200, 800
idx1 = pd.RangeIndex(N1); idx2 = pd.RangeIndex(N2)
X_full = pd.DataFrame({'close': np.random.randn(N1), 'volume': np.random.rand(N1)}, index=idx1)
y_full = pd.Series(np.random.randn(N1), index=idx1)

X_sample = X_full.iloc[:N2].copy()
y_sample = y_full.iloc[:N2].copy()

alpha_pool = [
    {'formula': 'BEG close END', 'ic': 0.02, 'weight': 1.0},
    {'formula': 'BEG volume END', 'ic': 0.03, 'weight': 1.0},
]
rc = RewardCalculator(alpha_pool)

# 终止奖励：在 sample 上合成 IC，不应抛异常
from mcts.environment import MDPState
s = MDPState(); s.add_token('BEG'); s.add_token('close'); s.add_token('END')
ic = rc.calculate_terminal_reward(s, X_sample, y_sample)
print("composite IC on sample:", ic)

# 没问题，输出结果是composite IC on sample: -0.5