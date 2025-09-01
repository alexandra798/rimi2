from collections import OrderedDict

import numpy as np
from scipy.stats import spearmanr, pearsonr
import logging
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.linear_model import Lasso

from utils.metrics import calculate_ic as _ic, calculate_ic
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
        self.utils_metrics = None
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
        # 统一走 utils 的对齐与 Spearman
        try:
            return float(_ic(predictions, targets, method='spearman'))
        except Exception as e:
            logger.error(f"Error calculating IC: {e}")
            return 0.0

    def calculate_daily_rank_ic(self, predictions, targets, X_like):
        import pandas as pd, numpy as np
        # 取出 date 索引
        if isinstance(X_like.index, pd.MultiIndex) and 'date' in X_like.index.names:
            dates = X_like.index.get_level_values('date')
        elif 'date' in getattr(X_like, 'columns', []):
            dates = X_like['date']
        else:
            return self.calculate_ic(predictions, targets)

        pred = pd.Series(getattr(predictions, 'values', predictions), index=targets.index)
        df = pd.concat([pred.rename('pred'), targets.rename('y'), dates.rename('date')], axis=1).dropna()
        if df.empty: return 0.0

        by_day = df.groupby('date').apply(lambda g: _ic(g['pred'], g['y'], method='spearman'))
        by_day = by_day.replace([np.inf, -np.inf], np.nan).dropna()
        return float(by_day.mean()) if len(by_day) else 0.0

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

    def _build_design_matrix(self, alpha_pool_dict, y_series):
        """
        alpha_pool_dict: { name: pd.Series(index=y.index) }
        返回对齐后的 X_mat (n, k) 与 y_vec (n,)
        """
        cols = []
        for k, s in alpha_pool_dict.items():
            s = pd.Series(s).rename(k)
            cols.append(s)
        df = pd.concat(cols + [pd.Series(y_series).rename("__y__")], axis=1).dropna()
        if df.empty:
            return None, None, []
        y_vec = df.pop("__y__").values.astype(float)
        names = list(df.columns)
        X_mat = df.values.astype(float)
        return X_mat, y_vec, names

    def _calculate_composite_ic(self, alpha_pool_dict, y_series, lasso_alpha=1e-3, max_iter=5000):
        """
        用 Lasso(fit_intercept=False) 合成；返回 (composite_series, coef_series)
        """
        X_mat, y_vec, names = self._build_design_matrix(alpha_pool_dict, y_series)
        if X_mat is None:  # 兜底
            return None, pd.Series(dtype=float)
        if X_mat.shape[1] == 0:
            return None, pd.Series(dtype=float)

        model = Lasso(alpha=lasso_alpha, fit_intercept=False, max_iter=max_iter)
        model.fit(X_mat, y_vec)
        coefs = pd.Series(model.coef_, index=names)

        # 稀疏：仅用非零系数做合成
        nz = (np.abs(coefs) > 1e-12)
        if nz.sum() == 0:
            # 全零兜底：退回到简单平均
            weights = pd.Series(1.0, index=names) / max(1, len(names))
        else:
            weights = coefs[nz]

        # 重新拼回索引对齐的复合因子
        parts = []
        for name, w in weights.items():
            parts.append(pd.Series(alpha_pool_dict[name]) * float(w))
        comp = None if not parts else pd.concat(parts, axis=1).sum(axis=1)
        return comp, coefs
    
    def _orthogonalized_gain(self, f_series, selected_mat_or_none, y_series):
        """
        计算“对已选集合正交后的边际 RankIC”
        f_series: pd.Series(index=y.index)
        selected_mat_or_none: np.ndarray of shape (n, k) or None
        y_series: pd.Series(index=f_series.index)
        """
        f = pd.concat([pd.Series(f_series).rename("f"), pd.Series(y_series).rename("y")], axis=1).dropna()
        if f.empty:
            return 0.0

        f_vec = f["f"].values.astype(float)
        y_vec = f["y"].values.astype(float)

        if selected_mat_or_none is None or selected_mat_or_none.shape[1] == 0:
            resid = f_vec
        else:
            coef, *_ = np.linalg.lstsq(selected_mat_or_none, f_vec, rcond=None)
            resid = f_vec - selected_mat_or_none @ coef

        # 残差与 y 的 Spearman
        r = pd.Series(resid, index=f.index)
        y = pd.Series(y_vec, index=f.index)
        try:
            return float(self.utils_metrics.calculate_ic(r, y, method='spearman'))
        except Exception:
            return 0.0