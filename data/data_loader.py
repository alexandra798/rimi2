"""数据加载和预处理模块"""
import pandas as pd
import numpy as np
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_user_dataset(file_path='price_volume_data_20100101_20250731.csv', target_column='target'):
    """
    加载用户数据集，设置目标列，并准备特征。

    Parameters:
    - file_path: 数据集文件路径 (CSV或.pt文件), 默认为 price_volume_data_20100101_20250731.csv
    - target_column: 数据集中的目标列名称, 默认为 'target'

    Returns:
    - X (特征), y (目标), all_features (特征名称列表)
    """
    logger.info(f"Loading dataset from {file_path}")

    # 判断文件类型
    if file_path.endswith('.pt'):
        # 加载PyTorch二进制文件
        data_dict = torch.load(file_path, weights_only=False)

        # 提取数据
        X_tensor = data_dict['X']
        y_tensor = data_dict['y']
        all_features = data_dict['feature_columns']

        # 转换为pandas DataFrame以保持与原代码的兼容性
        X = pd.DataFrame(X_tensor.numpy(), columns=all_features)
        y = pd.Series(y_tensor.numpy(), name=target_column)

        # 如果有date和ticker信息，重建索引
        if data_dict.get('has_date') and data_dict.get('has_ticker'):
            if 'dates' in data_dict and 'tickers' in data_dict:
                # 创建多级索引
                index = pd.MultiIndex.from_arrays(
                    [data_dict['tickers'], pd.to_datetime(data_dict['dates'])],
                    names=['ticker', 'date']
                )
                X.index = index
                y.index = index

    else:
        # 原始CSV加载逻辑
        user_dataset = pd.read_csv(file_path)

        # 转换日期列为datetime格式
        if 'date' in user_dataset.columns:
            user_dataset['date'] = pd.to_datetime(user_dataset['date'], errors='coerce')
            user_dataset.dropna(subset=['date'], inplace=True)
            # 如果存在ticker和date，设置多级索引
            if 'ticker' in user_dataset.columns:
                user_dataset.set_index(['ticker', 'date'], inplace=True)

        # 确保目标列存在
        if target_column not in user_dataset.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        # 分离特征和目标
        X = user_dataset.drop(columns=[target_column])
        y = user_dataset[target_column]

        # 获取特征名称列表
        all_features = X.columns.tolist()

    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y, all_features


def check_missing_values(dataset, dataset_name):
    """
    检查数据集中的缺失值

    Parameters:
    - dataset: 要检查缺失值的DataFrame
    - dataset_name: 数据集名称（用于打印）
    """
    missing_values = dataset.isnull().sum()
    missing_columns = missing_values[missing_values > 0]

    if not missing_columns.empty:
        logger.warning(f'Missing values in {dataset_name} dataset:')
        logger.warning(missing_columns)
    else:
        logger.info(f'No missing values in {dataset_name} dataset.')


def handle_missing_values(dataset, strategy='forward_fill', fill_value=0):
    """
    处理数据集中的缺失值
    
    Parameters:
    - dataset: 要处理的DataFrame
    - strategy: 处理策略 ('forward_fill', 'backward_fill', 'mean', 'median', 'zero', 'drop')
    - fill_value: 当策略为'zero'时使用的填充值
    
    Returns:
    - dataset: 处理后的DataFrame
    """
    if strategy == 'forward_fill':
        return dataset.ffill().fillna(fill_value)
    elif strategy == 'backward_fill':
        return dataset.bfill().fillna(fill_value)
    elif strategy == 'mean':
        return dataset.fillna(dataset.mean()).fillna(fill_value)
    elif strategy == 'median':
        return dataset.fillna(dataset.median()).fillna(fill_value)
    elif strategy == 'zero':
        return dataset.fillna(fill_value)
    elif strategy == 'drop':
        return dataset.dropna()
    else:
        logger.warning(f"Unknown strategy '{strategy}', using forward fill")
        return dataset.ffill().fillna(fill_value)


def apply_alphas_and_return_transformed(X, alpha_formulas, evaluate_formula_func):
    """
    应用顶级alpha公式到数据集，返回包含原始特征和新alpha特征的转换数据集

    Parameters:
    - X: 原始特征数据集
    - alpha_formulas: 要应用的alpha公式列表
    - evaluate_formula_func: 评估公式的函数

    Returns:
    - transformed_X: 包含原始特征和新alpha特征的数据集
    """
    transformed_X = X.copy()

    for formula in alpha_formulas:
        result = evaluate_formula_func(formula, X)
        # 处理结果中的NaN值
        result = result.fillna(0)
        transformed_X[formula] = result

    return transformed_X


def prepare_stock_features(raw_data):
    """
    准备论文要求的6个特征
    """
    features = pd.DataFrame()

    # 基础价格特征
    features['open'] = raw_data['open']
    features['high'] = raw_data['high']
    features['low'] = raw_data['low']
    features['close'] = raw_data['close']
    features['volume'] = raw_data['volume']

    # 计算VWAP (Volume Weighted Average Price)
    # VWAP = Σ(Price * Volume) / Σ(Volume)
    typical_price = (raw_data['high'] + raw_data['low'] + raw_data['close']) / 3
    features['vwap'] = (typical_price * raw_data['volume']).rolling(window=1).sum() / \
                       raw_data['volume'].rolling(window=1).sum()

    # 计算收益率目标
    returns_5d = raw_data['close'].pct_change(5).shift(-5)  # 未来5天收益率
    returns_10d = raw_data['close'].pct_change(10).shift(-10)  # 未来10天收益率

    return features, returns_5d, returns_10d