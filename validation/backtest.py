"""回测模块 - 修复版"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import logging


from alpha.evaluator import FormulaEvaluator

logger = logging.getLogger(__name__)


def backtest_formulas(formulas, X_test, y_test):
    """
    回测已发现的公式

    Parameters:
    - formulas: 要测试的公式列表
    - X_test: 测试特征数据
    - y_test: 测试目标数据

    Returns:
    - results: 公式及其IC值的字典
    """
    evaluator = FormulaEvaluator()
    results = {}

    for formula in formulas:
        # 使用统一的评估函数
        feature = evaluator.evaluate(formula, X_test)

        # 对齐数据
        valid_indices = ~(feature.isna() | y_test.isna())
        feature_clean = feature[valid_indices]
        y_test_clean = y_test[valid_indices]

        # 计算IC
        if len(feature_clean) > 1:
            ic, _ = spearmanr(feature_clean, y_test_clean)
            results[formula] = ic if not np.isnan(ic) else 0
        else:
            results[formula] = 0
            logger.warning(f"Insufficient data for formula: {formula}")

    return results


# validation/backtest.py 新增
def backtest_with_trading_simulation(formulas, X_test, y_test, price_data,
                                     top_k=40, rebalance_freq=5,
                                     initial_capital=1000000):
    """
    论文5.3节的完整交易模拟

    Parameters:
    - formulas: alpha公式列表
    - X_test: 测试特征数据
    - y_test: 实际收益率
    - price_data: 包含价格信息的DataFrame
    - top_k: 每次选择的股票数量
    - rebalance_freq: 重新平衡频率（天）
    - initial_capital: 初始资金
    """
    evaluator = FormulaEvaluator()

    # 获取时间索引（假设数据有date列）
    dates = X_test.index.get_level_values('date').unique() if isinstance(X_test.index,
                                                                         pd.MultiIndex) else X_test.index.unique()
    dates = sorted(dates)

    portfolio_values = [initial_capital]
    holdings = {}  # 当前持仓

    for i, date in enumerate(dates):
        # 每rebalance_freq天重新平衡
        if i % rebalance_freq == 0:
            # 计算所有股票的alpha信号
            daily_data = X_test.loc[X_test.index.get_level_values('date') == date]

            alpha_scores = {}
            for ticker in daily_data.index.get_level_values('ticker'):
                ticker_data = daily_data.loc[daily_data.index.get_level_values('ticker') == ticker]

                # 使用所有公式的平均信号
                scores = []
                for formula in formulas:
                    score = evaluator.evaluate(formula, ticker_data)
                    if not pd.isna(score).all():
                        scores.append(score.values[0] if hasattr(score, 'values') else score)

                if scores:
                    alpha_scores[ticker] = np.mean(scores)

            # 选择top-k股票
            if alpha_scores:
                sorted_tickers = sorted(alpha_scores.items(), key=lambda x: x[1], reverse=True)
                selected_tickers = [t[0] for t in sorted_tickers[:top_k]]

                # 计算每只股票的投资金额（等权重）
                current_value = portfolio_values[-1]
                position_size = current_value / len(selected_tickers)

                # 更新持仓
                new_holdings = {}
                for ticker in selected_tickers:
                    # 获取当前价格
                    current_price = price_data.loc[(date, ticker), 'close']
                    shares = position_size / current_price
                    new_holdings[ticker] = shares

                holdings = new_holdings

        # 计算当日组合价值
        daily_value = 0
        for ticker, shares in holdings.items():
            try:
                current_price = price_data.loc[(date, ticker), 'close']
                daily_value += shares * current_price
            except:
                # 股票可能停牌或退市
                pass

        if daily_value == 0:
            daily_value = portfolio_values[-1]  # 保持前一日价值

        portfolio_values.append(daily_value)

    # 计算性能指标
    portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # 累积收益率（论文公式）
    cumulative_return = (portfolio_values[-1] / portfolio_values[0]) - 1

    # 计算夏普比率
    sharpe_ratio = np.sqrt(252) * np.mean(portfolio_returns) / np.std(portfolio_returns)

    # 最大回撤
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max
    max_drawdown = np.min(drawdown)

    results = {
        'cumulative_return': cumulative_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': abs(max_drawdown),
        'portfolio_values': portfolio_values,
        'daily_returns': portfolio_returns
    }

    return results