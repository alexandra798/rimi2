# alpha/pool.py 完整重写
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
import logging


class AlphaPool:
    """论文Algorithm 1的完整实现"""

    def __init__(self, pool_size=100, lambda_param=0.1, learning_rate=0.01):
        self.pool_size = pool_size  # K
        self.lambda_param = lambda_param  # λ
        self.learning_rate = learning_rate

        self.alphas = []  # 存储 {formula, values, weight}
        self.model = None  # 线性组合模型

    def maintain_pool(self, new_alpha, X_data, y_data):
        """
        Algorithm 1: 维护alpha池
        输入：当前alpha集合F，新alpha f_new，组合模型c(·|F,ω)
        输出：最优alpha集合F*和权重ω*
        """
        # Step 1: F ← F ∪ f_new
        self.alphas.append(new_alpha)

        # Step 2-4: 梯度下降优化权重
        self._optimize_weights_gradient_descent(X_data, y_data)

        # Step 5-6: 如果超过池大小，移除权重最小的alpha
        if len(self.alphas) > self.pool_size:
            # 找到绝对权重最小的alpha
            min_weight_idx = np.argmin([abs(a.get('weight', 0)) for a in self.alphas])
            removed_alpha = self.alphas.pop(min_weight_idx)
            logging.info(f"Removed alpha with min weight: {removed_alpha['formula'][:50]}...")

            # 重新优化权重
            self._optimize_weights_gradient_descent(X_data, y_data)

        return self.alphas

    def _optimize_weights_gradient_descent(self, X_data, y_data, max_iters=100):
        """使用梯度下降优化权重（论文核心）"""
        if len(self.alphas) == 0:
            return

        # 构建特征矩阵
        feature_matrix = []
        for alpha in self.alphas:
            if 'values' in alpha:
                values = alpha['values']
                if hasattr(values, 'values'):
                    values = values.values
                feature_matrix.append(values.flatten())

        if not feature_matrix:
            return

        X = np.column_stack(feature_matrix)
        y = y_data.values if hasattr(y_data, 'values') else y_data

        # 确保维度匹配
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]

        # 移除NaN
        valid_mask = ~(np.any(np.isnan(X), axis=1) | np.isnan(y))
        if valid_mask.sum() < 10:
            return

        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        # 初始化权重
        weights = np.array([a.get('weight', 1.0 / len(self.alphas)) for a in self.alphas])

        # 梯度下降
        for iteration in range(max_iters):
            # 前向传播：计算预测值
            predictions = X_clean @ weights

            # 计算MSE损失
            error = predictions - y_clean
            loss = np.mean(error ** 2)

            # 反向传播：计算梯度
            gradient = 2.0 * (X_clean.T @ error) / len(y_clean)

            # 更新权重
            weights -= self.learning_rate * gradient

            # 早停条件
            if np.linalg.norm(gradient) < 1e-6:
                break

        # 更新alpha权重
        for i, alpha in enumerate(self.alphas):
            alpha['weight'] = weights[i]

        logging.debug(f"Optimized weights after {iteration + 1} iterations, loss={loss:.6f}")

    def get_composite_alpha_value(self, X_data):
        """计算合成alpha值"""
        if len(self.alphas) == 0:
            return None

        weighted_sum = None
        for alpha in self.alphas:
            if 'values' in alpha and 'weight' in alpha:
                values = alpha['values']
                weight = alpha['weight']

                if weighted_sum is None:
                    weighted_sum = weight * values
                else:
                    weighted_sum += weight * values

        return weighted_sum