# alpha/pool.py 完整修复版
import numpy as np
from sklearn.linear_model import LinearRegression
import logging
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


class AlphaPool:
    """论文Algorithm 1的完整实现 - 修复版"""

    def __init__(self, pool_size=100, lambda_param=0.1, learning_rate=0.01):
        self.pool_size = pool_size  # K
        self.lambda_param = lambda_param  # λ
        self.learning_rate = learning_rate

        self.alphas = []  # 存储 {formula, values, weight, ic, score}
        self.model = None  # 线性组合模型

    def add_to_pool(self, alpha_info):
        """
        添加单个alpha到池中

        Args:
            alpha_info: 字典，包含 'formula', 'score', 可选 'values', 'ic'
        """
        # 检查是否已存在
        if not any(a['formula'] == alpha_info['formula'] for a in self.alphas):
            # 确保有必要的字段
            if 'weight' not in alpha_info:
                alpha_info['weight'] = 1.0 / max(len(self.alphas), 1)
            if 'ic' not in alpha_info and 'score' in alpha_info:
                alpha_info['ic'] = alpha_info['score']

            self.alphas.append(alpha_info)
            logger.info(f"Added alpha to pool: {alpha_info['formula'][:50]}...")

            # 如果超过池大小，移除最差的
            if len(self.alphas) > self.pool_size:
                self._remove_worst_alpha()

    def update_pool(self, X_data, y_data, evaluate_formula):
        """
        更新整个池：评估所有公式并优化权重

        Args:
            X_data: 特征数据
            y_data: 目标数据
            evaluate_formula: 公式评估函数
        """
        logger.info(f"Updating alpha pool with {len(self.alphas)} formulas...")

        # 1. 评估所有没有values的alpha
        for alpha in self.alphas:
            if 'values' not in alpha or alpha['values'] is None:
                try:
                    alpha['values'] = evaluate_formula.evaluate(
                        alpha['formula'],
                        X_data,
                        allow_partial=False
                    )

                    # 计算IC
                    if alpha['values'] is not None and not alpha['values'].isna().all():
                        alpha['ic'] = self._calculate_ic(alpha['values'], y_data)
                    else:
                        alpha['ic'] = 0.0

                except Exception as e:
                    logger.warning(f"Failed to evaluate formula: {alpha['formula'][:50]}...")
                    alpha['values'] = None
                    alpha['ic'] = 0.0

        # 2. 移除无效的alpha
        self.alphas = [a for a in self.alphas if a.get('ic', 0) != 0]

        # 3. 优化权重
        if len(self.alphas) > 0:
            self._optimize_weights_gradient_descent(X_data, y_data)

        # 4. 根据IC排序
        self.alphas.sort(key=lambda x: abs(x.get('ic', 0)), reverse=True)

        # 5. 保持池大小
        if len(self.alphas) > self.pool_size:
            self.alphas = self.alphas[:self.pool_size]

        logger.info(f"Pool updated: {len(self.alphas)} valid alphas")

    def maintain_pool(self, new_alpha, X_data, y_data):
        """
        Algorithm 1: 维护alpha池（保留原有实现）
        输入：当前alpha集合F，新alpha f_new，组合模型c(·|F,ω)
        输出：最优alpha集合F*和权重ω*
        """
        # Step 1: F ← F ∪ f_new
        self.alphas.append(new_alpha)

        # Step 2-4: 梯度下降优化权重
        self._optimize_weights_gradient_descent(X_data, y_data)

        # Step 5-6: 如果超过池大小，移除权重最小的alpha
        if len(self.alphas) > self.pool_size:
            self._remove_worst_alpha()
            # 重新优化权重
            self._optimize_weights_gradient_descent(X_data, y_data)

        return self.alphas

    def get_top_formulas(self, n=5):
        """
        获取最佳的n个公式

        Args:
            n: 返回的公式数量

        Returns:
            公式字符串列表
        """
        # 根据IC或权重排序
        sorted_alphas = sorted(
            self.alphas,
            key=lambda x: abs(x.get('ic', 0) * x.get('weight', 1)),
            reverse=True
        )

        # 返回前n个公式
        top_formulas = []
        for alpha in sorted_alphas[:n]:
            formula = alpha['formula']
            ic = alpha.get('ic', 0)
            weight = alpha.get('weight', 1)
            logger.info(f"Top formula: {formula[:50]}... (IC={ic:.4f}, weight={weight:.4f})")
            top_formulas.append(formula)

        return top_formulas

    def _optimize_weights_gradient_descent(self, X_data, y_data, max_iters=100):
        """使用梯度下降优化权重（论文核心）"""
        if len(self.alphas) == 0:
            return

        # 构建特征矩阵
        feature_matrix = []
        valid_indices = []

        for i, alpha in enumerate(self.alphas):
            if 'values' in alpha and alpha['values'] is not None:
                values = alpha['values']
                if hasattr(values, 'values'):
                    values = values.values
                feature_matrix.append(values.flatten())
                valid_indices.append(i)

        if not feature_matrix:
            logger.warning("No valid alpha values for optimization")
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
            logger.warning("Insufficient valid data for optimization")
            return

        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        # 初始化权重
        weights = np.array([self.alphas[i].get('weight', 1.0 / len(valid_indices))
                            for i in valid_indices])

        # 梯度下降
        best_loss = float('inf')
        best_weights = weights.copy()

        for iteration in range(max_iters):
            # 前向传播：计算预测值
            predictions = X_clean @ weights

            # 计算MSE损失
            error = predictions - y_clean
            loss = np.mean(error ** 2)

            # 保存最佳权重
            if loss < best_loss:
                best_loss = loss
                best_weights = weights.copy()

            # 反向传播：计算梯度
            gradient = 2.0 * (X_clean.T @ error) / len(y_clean)

            # L2正则化
            gradient += self.lambda_param * weights

            # 更新权重
            weights -= self.learning_rate * gradient

            # 早停条件
            if np.linalg.norm(gradient) < 1e-6:
                break

        # 使用最佳权重更新alpha
        for idx, i in enumerate(valid_indices):
            self.alphas[i]['weight'] = best_weights[idx]

        logger.debug(f"Optimized weights after {iteration + 1} iterations, loss={best_loss:.6f}")

    def _remove_worst_alpha(self):
        """移除权重最小的alpha"""
        if len(self.alphas) > 0:
            # 找到绝对权重最小的alpha
            min_weight_idx = np.argmin([abs(a.get('weight', 0)) for a in self.alphas])
            removed_alpha = self.alphas.pop(min_weight_idx)
            logger.info(f"Removed alpha with min weight: {removed_alpha['formula'][:50]}...")

    def _calculate_ic(self, predictions, targets):
        """计算IC（Pearson相关系数）"""
        try:
            if hasattr(predictions, 'values'):
                predictions = predictions.values
            if hasattr(targets, 'values'):
                targets = targets.values

            predictions = np.array(predictions).flatten()
            targets = np.array(targets).flatten()

            # 对齐长度
            min_len = min(len(predictions), len(targets))
            predictions = predictions[:min_len]
            targets = targets[:min_len]

            # 移除NaN
            valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
            if valid_mask.sum() < 2:
                return 0.0

            corr, _ = pearsonr(predictions[valid_mask], targets[valid_mask])
            return corr if not np.isnan(corr) else 0.0

        except Exception as e:
            logger.error(f"Error calculating IC: {e}")
            return 0.0

    def get_composite_alpha_value(self, X_data):
        """计算合成alpha值"""
        if len(self.alphas) == 0:
            return None

        weighted_sum = None
        total_weight = 0

        for alpha in self.alphas:
            if 'values' in alpha and 'weight' in alpha and alpha['values'] is not None:
                values = alpha['values']
                weight = alpha['weight']

                if weighted_sum is None:
                    weighted_sum = weight * values
                else:
                    weighted_sum += weight * values

                total_weight += abs(weight)

        # 归一化
        if weighted_sum is not None and total_weight > 0:
            return weighted_sum / total_weight

        return weighted_sum

    def get_pool_statistics(self):
        """获取池的统计信息"""
        if not self.alphas:
            return {}

        ics = [a.get('ic', 0) for a in self.alphas]
        weights = [a.get('weight', 0) for a in self.alphas]

        return {
            'pool_size': len(self.alphas),
            'avg_ic': np.mean(ics),
            'max_ic': np.max(ics),
            'min_ic': np.min(ics),
            'avg_weight': np.mean(weights),
            'max_weight': np.max(weights),
            'min_weight': np.min(weights)
        }