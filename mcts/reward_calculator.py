from collections import OrderedDict

import numpy as np
from scipy.stats import spearmanr, pearsonr
import logging
from sklearn.linear_model import LinearRegression
import pandas as pd
from utils.metrics import calculate_ic
from core import RPNEvaluator,RPNValidator
from alpha.evaluator import FormulaEvaluator

logger = logging.getLogger(__name__)

class RewardCalculator:
    """
    - 中间奖励: Reward_inter = IC - λ * (1/k) * Σ mutIC_i
    - 终止奖励: Reward_end = 合成alpha的IC
    """

    def __init__(self, alpha_pool, lambda_param=0.1, sample_size=5000,
                 pool_size=100, min_std=1e-6, random_seed=42, cache_size=500):
        self.alpha_pool = alpha_pool
        self.lambda_param = lambda_param
        self.sample_size = sample_size
        self.pool_size = pool_size
        self.min_std = min_std  # 新增：最小标准差阈值
        self.random_seed = random_seed  # 保存seed
        self.formula_evaluator = FormulaEvaluator()
        self.cache_size = cache_size
        self._cache = OrderedDict()

        self.constant_penalty_count = 0

        self.rng = np.random.RandomState(random_seed)

    def _manage_cache(self):
        """管理缓存大小"""
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _finite_series(x, index=None):
        import numpy as np
        import pandas as pd
        s = pd.Series(x if not hasattr(x, 'values') else x.values, index=index)
        s = s.replace([np.inf, -np.inf], np.nan)
        return s[np.isfinite(s)]

    def is_nearly_constant(self, values):
        """检查值是否接近常数"""
        if values is None:
            return True

        if hasattr(values, 'values'):
            values = values.values
        values = np.array(values).flatten()

        valid_values = values[~np.isnan(values)]
        if len(valid_values) < 2:
            return True

        std = np.std(valid_values)
        return std < self.min_std

    def _build_design_matrix(self, formulas, X_ctx, y_ctx):
        """构建设计矩阵时确保数据对齐"""
        cols = []
        for f in formulas:
            s = self.formula_evaluator.evaluate(f, X_ctx, allow_partial=False)
            if s is None:
                continue
            if not isinstance(s, pd.Series):
                s = pd.Series(np.asarray(s).reshape(-1), index=X_ctx.index)
            s = s.replace([np.inf, -np.inf], np.nan).rename(f)
            cols.append(s)

        if not cols:
            return None, None

        # 使用内连接确保所有数据对齐
        df = pd.concat(cols + [y_ctx.rename('_y')], axis=1, join='inner').dropna()
        if df.shape[0] < 50 or df.shape[1] <= 1:
            return None, None

        X_mat = df.drop(columns=['_y']).to_numpy()
        y_vec = df['_y'].to_numpy()
        return X_mat, y_vec

    def calculate_intermediate_reward(self, state, X_data, y_data):
        cache_key = ' '.join([t.name for t in state.token_sequence])
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)  # LRU更新
            return self._cache[cache_key]

        if not RPNValidator.is_valid_partial_expression(state.token_sequence):
            return -0.1

        try:
            if len(X_data) > self.sample_size:
                sample_indices = self.rng.choice(len(X_data), self.sample_size, replace=False)
                X_sample = X_data.iloc[sample_indices] if hasattr(X_data, 'iloc') else X_data[sample_indices]
                y_sample = y_data.iloc[sample_indices] if hasattr(y_data, 'iloc') else y_data[sample_indices]
            else:
                X_sample = X_data
                y_sample = y_data

            # 评估
            alpha_values = self.formula_evaluator.evaluate_state(state, X_sample)

            if alpha_values is None or alpha_values.isna().all():
                return -0.1
            else:
                # 新增：检查是否为常数
                valid_values = alpha_values.dropna()

                if len(valid_values) > 10:
                    std = valid_values.std()
                    unique_ratio = len(valid_values.unique()) / len(valid_values)

                    # 多重检测
                    if std < self.min_std or unique_ratio < 0.01:
                        self.constant_penalty_count += 1
                        logger.debug(f"Constant alpha in intermediate state (std={std:.8f}, unique={unique_ratio:.2%})")
                        return -1.0  # 严厉惩罚


                # 计算IC
                ic = self.calculate_ic(alpha_values, y_sample)

                # 额外奖励高变异性的因子
                if hasattr(alpha_values, 'values'):
                    values = alpha_values.values
                else:
                    values = np.array(alpha_values)

                valid_values_for_bonus = values[~np.isnan(values)]
                if len(valid_values_for_bonus) > 0:
                    std = np.std(valid_values_for_bonus)
                    diversity_bonus = np.log(1 + std) * 0.1
                else:
                    diversity_bonus = 0

                # 计算mutIC
                if len(self.alpha_pool) > 0:
                    mut_ic_sum = 0
                    valid_count = 0
                    for alpha in self.alpha_pool[:10]:
                        if 'values' in alpha:
                            alpha_sample_values = self.formula_evaluator.evaluate(
                                alpha['formula'], X_sample
                            )
                            mut_ic = self._calculate_mutual_ic(alpha_values, alpha_sample_values)
                            if not np.isnan(mut_ic):
                                mut_ic_sum += abs(mut_ic)
                                valid_count += 1

                    if valid_count > 0:
                        avg_mut_ic = mut_ic_sum / valid_count
                        result = ic - self.lambda_param * avg_mut_ic + diversity_bonus
                    else:
                        result = ic + diversity_bonus
                else:
                    result = ic + diversity_bonus


            # 缓存结果
            self._cache[cache_key] = result
            self._manage_cache()
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
            formula_str = ' '.join([t.name for t in state.token_sequence])
            alpha_values = self.formula_evaluator.evaluate(
                formula_str,
                X_data,
                allow_partial=False
            )

            # 新增：严格检查常数
            if alpha_values is None or alpha_values.isna().all() or self.is_nearly_constant(alpha_values):
                logger.debug(f"Terminal state produces constant alpha: {formula_str[:50]}...")
                return -2.0  # 更严厉的惩罚

            # 计算个体IC
            individual_ic = self.calculate_ic(alpha_values, y_data)

            # 新增：IC太低则惩罚
            if abs(individual_ic) < 0.01:
                logger.debug(f"Terminal state has very low IC: {individual_ic:.4f}")
                return -0.5

            readable_formula = ' '.join([t.name for t in state.token_sequence])

            # 只添加高质量的alpha到池中
            if abs(individual_ic) >= 0.01 and not self.is_nearly_constant(alpha_values):
                new_alpha = {
                    'formula': readable_formula,
                    'values': alpha_values,
                    'ic': individual_ic,
                    'weight': 1.0
                }

                exists = any(a.get('formula') == readable_formula for a in self.alpha_pool)
                if not exists:
                    self.alpha_pool.append(new_alpha)

                    if len(self.alpha_pool) > self.pool_size:
                        self.alpha_pool.sort(key=lambda x: abs(x.get('ic', 0)), reverse=True)
                        self.alpha_pool = self.alpha_pool[:self.pool_size]

                    logger.info(f"High quality terminal alpha: {readable_formula[:50]}...")
                    logger.info(f"  IC={individual_ic:.4f}")

            # 计算合成IC
            composite_ic = self._calculate_composite_ic(X_data, y_data)

            return composite_ic

        except Exception as e:
            logger.error(f"Error in terminal reward: {e}")
            return -0.5



    def calculate_ic(self, predictions, targets):
        """计算IC（Pearson相关系数）-使用pandas对齐"""
        try:
            # 转为Series（如果不是）
            if not isinstance(predictions, pd.Series):
                if hasattr(predictions, 'values'):
                    predictions = pd.Series(predictions.values)
                else:
                    predictions = pd.Series(np.array(predictions).flatten())

            if not isinstance(targets, pd.Series):
                if hasattr(targets, 'values'):
                    targets = pd.Series(targets.values)
                else:
                    targets = pd.Series(np.array(targets).flatten())

            # ===== 使用pandas对齐 =====

            df = pd.concat([predictions.rename('pred'), targets.rename('target')],
                           axis=1, join='inner')
            df = df.replace([np.inf, -np.inf], np.nan).dropna()

            if len(df) < 2:
                return 0.0

            # 常数检测
            if df['pred'].std() < self.min_std or df['target'].std() < self.min_std:
                logger.debug("IC calculation skipped: constant values detected")
                return 0.0

            corr, _ = pearsonr(df['pred'], df['target'])

            return float(corr) if not np.isnan(corr) else 0.0

        except Exception as e:
            logger.error(f"Error calculating IC: {e}")
            return 0.0

    def _calculate_mutual_ic(self, alpha1_values, alpha2_values):
        """计算两个alpha的相互IC"""
        try:
            # 转为Series
            if not isinstance(alpha1_values, pd.Series):
                alpha1_values = pd.Series(getattr(alpha1_values, 'values', alpha1_values))
            if not isinstance(alpha2_values, pd.Series):
                alpha2_values = pd.Series(getattr(alpha2_values, 'values', alpha2_values))

            # ===== 使用pandas对齐 =====
            df = pd.concat([alpha1_values.rename('a1'), alpha2_values.rename('a2')],
                           axis=1, join='inner')
            df = df.replace([np.inf, -np.inf], np.nan).dropna()

            if len(df) < 2:
                return 0.0

            # 常数检测
            if df['a1'].std() < self.min_std or df['a2'].std() < self.min_std:
                return 0.0

            corr, _ = pearsonr(df['a1'], df['a2'])
            return float(corr) if not np.isnan(corr) else 0.0

        except Exception as e:
            logger.error(f"Error calculating mutual IC: {e}")
            return 0.0

    def _calculate_composite_ic(self, X_data, y_data):
        if len(self.alpha_pool) == 0:
            return 0.0

            # 仅取有公式字段的
        formulas = [a['formula'] for a in self.alpha_pool if 'formula' in a]
        X_mat, y_vec = self._build_design_matrix(formulas, X_data, y_data)
        if X_mat is None:
            # 回退：用池内 alpha 的平均 IC
            valid_ic = [a.get('ic', 0.0) for a in self.alpha_pool if 'ic' in a]
            return float(np.mean(valid_ic)) if valid_ic else 0.0

        self.linear_model = LinearRegression(fit_intercept=False)
        self.linear_model.fit(X_mat, y_vec)

        # 同步权重（可选）
        weights = self.linear_model.coef_
        for i, a in enumerate(self.alpha_pool):
            if i < len(weights):
                a['weight'] = float(weights[i])

        pred = self.linear_model.predict(X_mat)
        return float(calculate_ic(pred, y_vec))
