"""所有操作符的集中实现"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, skew, kurtosis
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class Operators:
    """所有操作符的静态方法集合"""

    @staticmethod
    def safe_divide(x, y, default_value=0):
        """安全除法函数，避免除零错误"""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(np.abs(y) < 1e-8, default_value, x / y)
            result = np.where(np.isnan(result) | np.isinf(result), default_value, result)
        return result

    # 基础操作符
    @staticmethod
    def ts_ref(x, t):
        """Ref operator: t天前的值"""
        if isinstance(x, pd.Series):
            return x.shift(t)
        elif isinstance(x, pd.DataFrame):
            return x.shift(t)
        else:
            raise TypeError("ref operator requires pandas Series or DataFrame")

    @staticmethod
    def csrank(x):
        """CSRank operator: 横截面排名"""
        if isinstance(x, pd.Series):
            if isinstance(x.index, pd.MultiIndex):
                return x.groupby(level=1).rank(pct=True)
            else:
                return x.rank(pct=True)
        else:
            raise TypeError("csrank operator requires pandas Series")

    # 一元操作符
    @staticmethod
    def sign(x):
        """Sign operator: 返回1如果x为正，否则返回0"""
        return np.where(x > 0, 1, 0)

    @staticmethod
    def abs_op(x):
        """Abs operator: 绝对值"""
        return np.abs(x)

    @staticmethod
    def log(x):
        """Log operator: 自然对数"""
        return np.where(x > 0, np.log(x), np.nan)

    # 比较操作符
    @staticmethod
    def greater(x, y):
        """Greater operator: x > y返回1，否则0"""
        return np.where(x > y, 1, 0)

    @staticmethod
    def less(x, y):
        """Less operator: x < y返回1，否则0"""
        return np.where(x < y, 1, 0)

    # 时序操作符
    @staticmethod
    def ts_rank(x, t):
        """Rank operator: 当前值在过去t天中的排名"""
        if isinstance(x, pd.Series):
            return x.rolling(window=t, min_periods=1).apply(
                lambda w: pd.Series(w).rank(pct=True).iloc[-1]
            )
        else:
            raise TypeError("rank operator requires pandas Series")

    @staticmethod
    def std(x, t):
        """Std operator: 过去t天的标准差"""
        if isinstance(x, pd.Series):
            return x.rolling(window=t, min_periods=1).std().fillna(0)
        else:
            raise TypeError("std operator requires pandas Series")

    @staticmethod
    def ts_max(x, t):
        """Max operator: 过去t天的最大值"""
        if isinstance(x, pd.Series):
            return x.rolling(window=t, min_periods=1).max()
        else:
            raise TypeError("ts_max operator requires pandas Series")

    @staticmethod
    def ts_min(x, t):
        """Min operator: 过去t天的最小值"""
        if isinstance(x, pd.Series):
            return x.rolling(window=t, min_periods=1).min()
        else:
            raise TypeError("ts_min operator requires pandas Series")

    @staticmethod
    def skew(x, t):
        """Skew operator: 过去t天的偏度"""
        if isinstance(x, pd.Series):
            return x.rolling(window=t, min_periods=1).skew().fillna(0)
        else:
            raise TypeError("skew operator requires pandas Series")

    @staticmethod
    def kurt(x, t):
        """Kurt operator: 过去t天的峰度"""
        if isinstance(x, pd.Series):
            return x.rolling(window=t, min_periods=1).kurt().fillna(0)
        else:
            raise TypeError("kurt operator requires pandas Series")

    @staticmethod
    def mean(x, t):
        """Mean operator: 过去t天的平均值"""
        if isinstance(x, pd.Series):
            return x.rolling(window=t, min_periods=1).mean().fillna(0)
        else:
            raise TypeError("mean operator requires pandas Series")

    @staticmethod
    def med(x, t):
        """Med operator: 过去t天的中位数"""
        if isinstance(x, pd.Series):
            return x.rolling(window=t, min_periods=1).median()
        else:
            raise TypeError("med operator requires pandas Series")

    @staticmethod
    def ts_sum(x, t):
        """Sum operator: 过去t天的总和"""
        if isinstance(x, pd.Series):
            return x.rolling(window=t, min_periods=1).sum()
        else:
            raise TypeError("ts_sum operator requires pandas Series")

    @staticmethod
    def cov(x, y, t):
        """Cov operator: 两个特征在过去t天的协方差"""
        if isinstance(x, pd.Series) and isinstance(y, pd.Series):
            return x.rolling(window=t, min_periods=1).cov(y)
        else:
            raise TypeError("cov operator requires two pandas Series")

    @staticmethod
    def corr(x, y, t):
        """Corr operator: 两个特征在过去t天的相关系数"""
        if isinstance(x, pd.Series) and isinstance(y, pd.Series):
            return x.rolling(window=t, min_periods=1).corr(y)
        else:
            raise TypeError("corr operator requires two pandas Series")

    @staticmethod
    def decay_linear(x, t):
        """Decay_linear operator: 线性衰减加权移动平均"""
        if isinstance(x, pd.Series):
            weights = np.arange(1, t + 1)
            weights = weights / weights.sum()
            result = x.rolling(window=t, min_periods=1).apply(
                lambda w: np.dot(w[~np.isnan(w)], weights[:len(w[~np.isnan(w)])])
                if len(w[~np.isnan(w)]) > 0 else 0
            )
            result = result.replace([np.inf, -np.inf], np.nan)
            return result.fillna(0)
        else:
            raise TypeError("decay_linear operator requires pandas Series")

    @staticmethod
    def wma(x, t):
        """WMA operator: 加权移动平均"""
        if isinstance(x, pd.Series):
            weights = np.arange(1, t + 1)
            weights = weights / weights.sum()
            result = x.rolling(window=t, min_periods=1).apply(
                lambda w: np.dot(w[~np.isnan(w)], weights[:len(w[~np.isnan(w)])])
                if len(w[~np.isnan(w)]) > 0 else 0
            )
            result = result.replace([np.inf, -np.inf], np.nan)
            return result.fillna(0)
        else:
            raise TypeError("wma operator requires pandas Series")

    @staticmethod
    def ema(x, t):
        """EMA operator: 指数移动平均"""
        if isinstance(x, pd.Series):
            return x.ewm(span=t, adjust=False).mean()
        else:
            raise TypeError("ema operator requires pandas Series")