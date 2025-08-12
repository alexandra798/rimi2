import numpy as np
from utils.metrics import calculate_sharpe_ratio

r = np.array([0.01, np.nan, 0.02, np.inf, -0.03, 0.0])
print("Sharpe:", calculate_sharpe_ratio(r))
