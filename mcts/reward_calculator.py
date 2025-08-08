import numpy as np
from scipy.stats import spearmanr, pearsonr
import logging
from sklearn.linear_model import LinearRegression

from core import RPNEvaluator,RPNValidator
from alpha import FormulaEvaluator

logger = logging.getLogger(__name__)

class RewardCalculator:
    """
    - 中间奖励: Reward_inter = IC - λ * (1/k) * Σ mutIC_i
    - 终止奖励: Reward_end = 合成alpha的IC
    """

    def __init__(self, alpha_pool, lambda_param=0.1, sample_size=5000,pool_size=100):
        self.alpha_pool = alpha_pool
        self.lambda_param = lambda_param
        self.sample_size = sample_size  # 采样大小
        self.pool_size = pool_size
        self.formula_evaluator = FormulaEvaluator()
        self._cache = {}  # 添加缓存

    def calculate_ic(self, predictions, targets):
        return self._calculate_ic(predictions, targets)

    def calculate_intermediate_reward(self, state, X_data, y_data):
        # 生成缓存键
        cache_key = ' '.join([t.name for t in state.token_sequence])
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 检查是否为合法的部分表达式
        if not RPNValidator.is_valid_partial_expression(state.token_sequence):
            return -0.1

        try:
            # 采样数据以加速计算
            if len(X_data) > self.sample_size:
                sample_indices = np.random.choice(len(X_data), self.sample_size, replace=False)
                X_sample = X_data.iloc[sample_indices] if hasattr(X_data, 'iloc') else X_data[sample_indices]
                y_sample = y_data.iloc[sample_indices] if hasattr(y_data, 'iloc') else y_data[sample_indices]
            else:
                X_sample = X_data
                y_sample = y_data

            # 评估
            alpha_values = self.formula_evaluator.evaluate_state(state, X_sample)

            if alpha_values is None:
                result = -0.1
            else:
                # 计算IC
                ic = self._calculate_ic(alpha_values, y_sample)

                # 计算mutIC（也用采样数据）
                if len(self.alpha_pool) > 0:
                    mut_ic_sum = 0
                    valid_count = 0
                    for alpha in self.alpha_pool[:10]:  # 只比较前10个alpha
                        if 'values' in alpha:
                            # 重新评估alpha在采样数据上的值
                            alpha_sample_values = self.formula_evaluator.evaluate(
                                alpha['formula'], X_sample
                            )
                            mut_ic = self._calculate_mutual_ic(alpha_values, alpha_sample_values)
                            if not np.isnan(mut_ic):
                                mut_ic_sum += abs(mut_ic)
                                valid_count += 1

                    if valid_count > 0:
                        avg_mut_ic = mut_ic_sum / valid_count
                        result = ic - self.lambda_param * avg_mut_ic
                    else:
                        result = ic
                else:
                    result = ic

            # 缓存结果
            self._cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"Error in intermediate reward: {e}")
            return -0.1

    def calculate_terminal_reward(self, state, X_data, y_data, evaluate_func=None):
        """计算终止奖励（合成alpha的IC）"""
        # 验证是否正确终止
        if state.token_sequence[-1].name != 'END':
            return -1.0

        try:
            # 使用统一的评估器
            formula_str = ' '.join([t.name for t in state.token_sequence])
            alpha_values = self.formula_evaluator.evaluate(
                formula_str,
                X_data,
                allow_partial=False
            )

            if alpha_values is None or alpha_values.isna().all():
                return -0.5

            # 计算个体IC
            individual_ic = self._calculate_ic(alpha_values, y_data)

            readable_formula = ' '.join([t.name for t in state.token_sequence])

            # 添加到池中
            new_alpha = {
                'formula': readable_formula,
                'values': alpha_values,
                'ic': individual_ic,
                'weight': 1.0
            }

            # 检查是否已存在
            exists = any(a.get('formula') == readable_formula for a in self.alpha_pool)
            if not exists:
                self.alpha_pool.append(new_alpha)

                # 维护池大小 - 使用self.pool_size
                if len(self.alpha_pool) > self.pool_size:
                    self.alpha_pool.sort(key=lambda x: abs(x.get('ic', 0)), reverse=True)
                    self.alpha_pool = self.alpha_pool[:self.pool_size]

                    # 计算合成IC
            composite_ic = self._calculate_composite_ic(y_data)

            logger.info(f"Terminal: formula={readable_formula[:50]}...")
            logger.info(f"  individual_IC={individual_ic:.4f}, composite_IC={composite_ic:.4f}")

            return composite_ic

        except Exception as e:
            logger.error(f"Error in terminal reward: {e}")
            return -0.5



    def _calculate_ic(self, predictions, targets):
        """计算IC（Pearson相关系数）- 修正版"""
        try:
            # 处理predictions
            if hasattr(predictions, 'values'):
                predictions = predictions.values
            if hasattr(targets, 'values'):
                targets = targets.values

            # 确保是numpy数组
            predictions = np.array(predictions).flatten()
            targets = np.array(targets).flatten()

            # 检查长度
            if len(predictions) == 1 and len(targets) > 1:
                # predictions是标量，扩展为向量
                predictions = np.full(len(targets), predictions[0])
            elif len(targets) == 1 and len(predictions) > 1:
                # targets是标量（不应该发生）
                logger.error("Targets is scalar, this should not happen")
                return 0.0
            elif len(predictions) != len(targets):
                # 长度不匹配，取最小长度
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

    def _calculate_mutual_ic(self, alpha1_values, alpha2_values):
        """计算两个alpha的相互IC"""
        try:
            if hasattr(alpha1_values, 'values'):
                alpha1_values = alpha1_values.values
            if hasattr(alpha2_values, 'values'):
                alpha2_values = alpha2_values.values

            alpha1 = np.array(alpha1_values).flatten()
            alpha2 = np.array(alpha2_values).flatten()

            min_len = min(len(alpha1), len(alpha2))
            alpha1 = alpha1[:min_len]
            alpha2 = alpha2[:min_len]

            valid_mask = ~(np.isnan(alpha1) | np.isnan(alpha2))
            if valid_mask.sum() < 2:
                return 0.0

            corr, _ = pearsonr(alpha1[valid_mask], alpha2[valid_mask])
            return corr if not np.isnan(corr) else 0.0

        except Exception as e:
            logger.error(f"Error calculating mutual IC: {e}")
            return 0.0

    def _calculate_composite_ic(self, y_data):
        """
        计算合成alpha的IC（论文Algorithm 1）
        使用线性回归组合所有alpha
        """
        if len(self.alpha_pool) == 0:
            return 0.0

        try:
            # 筛选有效alpha
            valid_alphas = [a for a in self.alpha_pool
                            if 'values' in a and a['values'] is not None]

            if len(valid_alphas) == 0:
                return 0.0

            # 如果只有一个alpha，直接返回其IC
            if len(valid_alphas) == 1:
                return valid_alphas[0].get('ic', 0)

            # 构建特征矩阵
            feature_matrix = []
            for alpha in valid_alphas:
                values = alpha['values']
                if hasattr(values, 'values'):
                    values = values.values
                feature_matrix.append(np.array(values).flatten())

            feature_matrix = np.column_stack(feature_matrix)

            # 准备目标数据
            if hasattr(y_data, 'values'):
                y_array = y_data.values
            else:
                y_array = np.array(y_data).flatten()

            # 对齐长度
            min_len = min(len(feature_matrix), len(y_array))
            feature_matrix = feature_matrix[:min_len]
            y_array = y_array[:min_len]

            # 移除NaN
            valid_mask = ~(np.any(np.isnan(feature_matrix), axis=1) | np.isnan(y_array))

            if valid_mask.sum() < 10:
                # 数据太少，返回平均IC
                return np.mean([a.get('ic', 0) for a in valid_alphas])

            # 训练线性模型（论文的核心）
            self.linear_model = LinearRegression(fit_intercept=False)
            self.linear_model.fit(feature_matrix[valid_mask], y_array[valid_mask])

            # 更新权重
            weights = self.linear_model.coef_
            for i, alpha in enumerate(valid_alphas):
                if i < len(weights):
                    alpha['weight'] = weights[i]

            # 计算合成预测
            composite_predictions = self.linear_model.predict(feature_matrix[valid_mask])

            # 计算合成IC
            composite_ic = self._calculate_ic(composite_predictions, y_array[valid_mask])

            return composite_ic

        except Exception as e:
            logger.error(f"Error in composite IC: {e}")
            return np.mean([a.get('ic', 0) for a in self.alpha_pool if 'ic' in a])