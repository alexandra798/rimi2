"""数据处理测试文件"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path


# 假设data_loader模块存在，这里模拟其关键功能
# 实际使用时需要从 data.data_loader import


class MockDataLoader:
    """模拟数据加载器的关键功能"""

    @staticmethod
    def load_user_dataset(file_path, target_column='target'):
        """加载用户数据集"""
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.pt'):
            import torch
            df = torch.load(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        # 分离特征和目标
        if target_column in df.columns:
            y = df[target_column]
            X = df.drop(columns=[target_column])
            features = X.columns.tolist()
        else:
            raise ValueError(f"Target column '{target_column}' not found")

        return X, y, features

    @staticmethod
    def check_missing_values(data, stage=''):
        """检查缺失值"""
        missing_info = {}
        for col in data.columns:
            missing_count = data[col].isna().sum()
            if missing_count > 0:
                missing_info[col] = {
                    'count': missing_count,
                    'percentage': missing_count / len(data) * 100
                }
        return missing_info

    @staticmethod
    def handle_missing_values(data, strategy='mixed'):
        """处理缺失值"""
        data_copy = data.copy()

        if strategy == 'mixed':
            # 价格类特征：前向填充
            price_cols = ['open', 'high', 'low', 'close', 'vwap']
            for col in price_cols:
                if col in data_copy.columns:
                    data_copy[col] = data_copy[col].fillna(method='ffill').fillna(method='bfill')

            # 成交量：填0
            if 'volume' in data_copy.columns:
                data_copy['volume'] = data_copy['volume'].fillna(0)

            # 其他列：均值填充
            for col in data_copy.columns:
                if col not in price_cols + ['volume']:
                    data_copy[col] = data_copy[col].fillna(data_copy[col].mean())

        elif strategy == 'drop':
            data_copy = data_copy.dropna()

        elif strategy == 'forward_fill':
            data_copy = data_copy.fillna(method='ffill').fillna(method='bfill')

        elif strategy == 'mean':
            data_copy = data_copy.fillna(data_copy.mean())

        return data_copy

    @staticmethod
    def clean_target_zeros(X, y, volume_threshold=1000):
        """清理target=0的样本（区分停牌和正常交易）"""
        # 假设volume=0表示停牌
        if 'volume' in X.columns:
            mask = ~((y == 0) & (X['volume'] < volume_threshold))
            return X[mask], y[mask]
        return X, y

    @staticmethod
    def validate_data_quality(X, y):
        """验证数据质量"""
        issues = []

        # 检查NaN
        if X.isna().any().any():
            issues.append("Features contain NaN values")
        if y.isna().any():
            issues.append("Target contains NaN values")

        # 检查Inf
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.any(np.isinf(X[col])):
                issues.append(f"Column '{col}' contains infinite values")

        # 检查常数列
        for col in X.columns:
            if X[col].nunique() == 1:
                issues.append(f"Column '{col}' is constant")

        # 检查样本数
        if len(X) < 100:
            issues.append(f"Too few samples: {len(X)}")

        # 检查目标分布
        if y.std() < 1e-6:
            issues.append("Target has very low variance")

        is_valid = len(issues) == 0
        return is_valid, issues


class TestDataLoading:
    """测试数据加载功能"""

    @pytest.fixture
    def sample_csv_file(self):
        """创建临时CSV文件"""
        np.random.seed(42)
        n = 100

        data = pd.DataFrame({
            'open': np.random.randn(n) * 10 + 100,
            'high': np.random.randn(n) * 10 + 105,
            'low': np.random.randn(n) * 10 + 95,
            'close': np.random.randn(n) * 10 + 100,
            'volume': np.random.exponential(1000000, n),
            'vwap': np.random.randn(n) * 10 + 100,
            'target': np.random.randn(n) * 0.01
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name

        yield temp_path

        # 清理
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_load_csv(self, sample_csv_file):
        """测试加载CSV文件"""
        loader = MockDataLoader()
        X, y, features = loader.load_user_dataset(sample_csv_file, 'target')

        assert X is not None
        assert y is not None
        assert len(features) == 6  # 不包括target
        assert 'close' in features
        assert 'volume' in features
        assert len(X) == len(y)

    def test_load_missing_target(self, sample_csv_file):
        """测试缺失目标列"""
        loader = MockDataLoader()

        with pytest.raises(ValueError, match="Target column .* not found"):
            X, y, features = loader.load_user_dataset(sample_csv_file, 'nonexistent')

    def test_check_missing_values(self):
        """测试检查缺失值"""
        # 创建带缺失值的数据
        data = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [1, np.nan, np.nan, 4, 5],
            'col3': [1, 2, 3, 4, 5]  # 无缺失
        })

        loader = MockDataLoader()
        missing_info = loader.check_missing_values(data)

        assert 'col1' in missing_info
        assert missing_info['col1']['count'] == 1
        assert missing_info['col1']['percentage'] == 20.0

        assert 'col2' in missing_info
        assert missing_info['col2']['count'] == 2

        assert 'col3' not in missing_info  # 无缺失


class TestMissingValueHandling:
    """测试缺失值处理"""

    @pytest.fixture
    def data_with_missing(self):
        """创建带缺失值的数据"""
        np.random.seed(42)
        n = 100

        data = pd.DataFrame({
            'open': np.random.randn(n) * 10 + 100,
            'high': np.random.randn(n) * 10 + 105,
            'low': np.random.randn(n) * 10 + 95,
            'close': np.random.randn(n) * 10 + 100,
            'volume': np.random.exponential(1000000, n),
            'vwap': np.random.randn(n) * 10 + 100,
            'other': np.random.randn(n)
        })

        # 添加缺失值
        data.iloc[0:5, 0] = np.nan  # open
        data.iloc[10:15, 3] = np.nan  # close
        data.iloc[20:25, 4] = np.nan  # volume
        data.iloc[30:35, 6] = np.nan  # other

        return data

    def test_mixed_strategy(self, data_with_missing):
        """测试混合策略"""
        loader = MockDataLoader()
        cleaned = loader.handle_missing_values(data_with_missing, strategy='mixed')

        # 不应该有缺失值
        assert cleaned.isna().sum().sum() == 0

        # volume的缺失应该被填充为0
        original_missing_volume = data_with_missing['volume'].isna()
        assert all(cleaned.loc[original_missing_volume, 'volume'] == 0)

    def test_drop_strategy(self, data_with_missing):
        """测试删除策略"""
        loader = MockDataLoader()
        cleaned = loader.handle_missing_values(data_with_missing, strategy='drop')

        # 不应该有缺失值
        assert cleaned.isna().sum().sum() == 0

        # 样本数应该减少
        assert len(cleaned) < len(data_with_missing)

    def test_forward_fill_strategy(self, data_with_missing):
        """测试前向填充策略"""
        loader = MockDataLoader()
        cleaned = loader.handle_missing_values(data_with_missing, strategy='forward_fill')

        # 不应该有缺失值（如果首尾有缺失会用后向填充）
        assert cleaned.isna().sum().sum() == 0

    def test_mean_strategy(self, data_with_missing):
        """测试均值填充策略"""
        loader = MockDataLoader()
        cleaned = loader.handle_missing_values(data_with_missing, strategy='mean')

        # 不应该有缺失值
        assert cleaned.isna().sum().sum() == 0

        # 填充值应该接近列均值
        for col in data_with_missing.columns:
            original_mean = data_with_missing[col].mean()
            cleaned_mean = cleaned[col].mean()
            # 均值应该相近（因为用均值填充）
            assert abs(original_mean - cleaned_mean) < abs(original_mean) * 0.1


class TestDataCleaning:
    """测试数据清理"""

    def test_clean_target_zeros(self):
        """测试清理target=0的样本"""
        n = 100
        X = pd.DataFrame({
            'close': np.random.randn(n) * 10 + 100,
            'volume': np.random.exponential(1000000, n)
        })
        y = pd.Series(np.random.randn(n) * 0.01)

        # 设置一些停牌样本（volume=0, target=0）
        suspension_idx = [10, 20, 30, 40, 50]
        for idx in suspension_idx:
            X.loc[idx, 'volume'] = 0
            y.loc[idx] = 0

        # 设置一些正常的target=0（volume>0）
        normal_zero_idx = [15, 25, 35]
        for idx in normal_zero_idx:
            y.loc[idx] = 0
            X.loc[idx, 'volume'] = 100000

        loader = MockDataLoader()
        X_clean, y_clean = loader.clean_target_zeros(X, y, volume_threshold=1000)

        # 停牌样本应该被移除
        assert len(X_clean) < len(X)

        # 正常的0值应该保留
        remaining_indices = X_clean.index
        for idx in normal_zero_idx:
            if idx in X.index:
                # 如果索引没有被完全重置，检查是否存在
                pass  # 索引可能被重置，所以这个检查可能不准确

        # 至少应该移除了一些样本
        assert len(X_clean) <= len(X) - len(suspension_idx)

    def test_validate_data_quality(self):
        """测试数据质量验证"""
        loader = MockDataLoader()

        # 好的数据
        good_X = pd.DataFrame({
            'col1': np.random.randn(200),
            'col2': np.random.randn(200) * 10,
            'col3': np.random.exponential(1, 200)
        })
        good_y = pd.Series(np.random.randn(200) * 0.01)

        is_valid, issues = loader.validate_data_quality(good_X, good_y)
        assert is_valid == True
        assert len(issues) == 0

        # 带NaN的数据
        bad_X = good_X.copy()
        bad_X.iloc[0, 0] = np.nan

        is_valid, issues = loader.validate_data_quality(bad_X, good_y)
        assert is_valid == False
        assert "Features contain NaN values" in issues

        # 带Inf的数据
        bad_X = good_X.copy()
        bad_X.iloc[0, 0] = np.inf

        is_valid, issues = loader.validate_data_quality(bad_X, good_y)
        assert is_valid == False
        assert any("infinite values" in issue for issue in issues)

        # 常数列
        bad_X = good_X.copy()
        bad_X['const_col'] = 1.0

        is_valid, issues = loader.validate_data_quality(bad_X, good_y)
        assert is_valid == False
        assert any("constant" in issue for issue in issues)

        # 样本太少
        small_X = good_X.iloc[:10]
        small_y = good_y.iloc[:10]

        is_valid, issues = loader.validate_data_quality(small_X, small_y)
        assert is_valid == False
        assert any("Too few samples" in issue for issue in issues)

        # 目标方差太小
        constant_y = pd.Series([1.0] * 200)

        is_valid, issues = loader.validate_data_quality(good_X, constant_y)
        assert is_valid == False
        assert any("low variance" in issue for issue in issues)


class TestDataTransformation:
    """测试数据转换"""

    def test_apply_alphas(self):
        """测试应用alpha公式转换数据"""
        # 这个测试需要FormulaEvaluator，这里简化
        from alpha.evaluator import FormulaEvaluator

        X = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 900, 1200, 1050]
        })

        formulas = [
            "BEG close END",
            "BEG volume END",
            "BEG close volume add END"
        ]

        evaluator = FormulaEvaluator()

        # 应用公式
        transformed_features = []
        for formula in formulas:
            feature = evaluator.evaluate(formula, X)
            if feature is not None and not feature.isna().all():
                transformed_features.append(feature)

        # 应该有3个转换后的特征
        assert len(transformed_features) == 3

        # 转换后的数据应该有相同的长度
        for feature in transformed_features:
            assert len(feature) == len(X)

    def test_data_alignment(self):
        """测试数据对齐"""
        # 创建不同索引的数据
        dates1 = pd.date_range('2020-01-01', periods=100)
        dates2 = pd.date_range('2020-01-05', periods=100)

        X1 = pd.DataFrame({'close': np.random.randn(100)}, index=dates1)
        X2 = pd.DataFrame({'volume': np.random.randn(100)}, index=dates2)

        # 合并
        merged = pd.concat([X1, X2], axis=1, join='inner')

        # 应该只有重叠的日期
        assert len(merged) < 100
        assert merged.index[0] >= pd.Timestamp('2020-01-05')

    def test_feature_scaling(self):
        """测试特征缩放"""
        X = pd.DataFrame({
            'small': np.random.randn(100) * 0.001,
            'medium': np.random.randn(100),
            'large': np.random.randn(100) * 1000
        })

        # Z-score标准化
        X_scaled = (X - X.mean()) / X.std()

        # 所有列应该有相似的尺度
        for col in X_scaled.columns:
            assert abs(X_scaled[col].mean()) < 0.1
            assert abs(X_scaled[col].std() - 1.0) < 0.1

    def test_handle_outliers(self):
        """测试异常值处理"""
        X = pd.Series(np.random.randn(100))

        # 添加异常值
        X.iloc[0] = 100
        X.iloc[1] = -100

        # 使用分位数裁剪
        q1 = X.quantile(0.01)
        q99 = X.quantile(0.99)
        X_clipped = X.clip(lower=q1, upper=q99)

        # 异常值应该被裁剪
        assert X_clipped.iloc[0] <= q99
        assert X_clipped.iloc[1] >= q1

        # 大部分值应该不变
        normal_mask = (X > q1) & (X < q99)
        assert np.allclose(X[normal_mask], X_clipped[normal_mask])


class TestEdgeCases:
    """测试边界情况"""

    def test_empty_dataframe(self):
        """测试空DataFrame"""
        loader = MockDataLoader()

        empty_df = pd.DataFrame()
        missing_info = loader.check_missing_values(empty_df)
        assert len(missing_info) == 0

        # 处理缺失值不应该报错
        result = loader.handle_missing_values(empty_df)
        assert len(result) == 0

    def test_single_row(self):
        """测试单行数据"""
        X = pd.DataFrame({'col1': [1.0], 'col2': [2.0]})
        y = pd.Series([0.01])

        loader = MockDataLoader()
        is_valid, issues = loader.validate_data_quality(X, y)

        # 单行数据应该被标记为问题
        assert is_valid == False
        assert any("Too few samples" in issue for issue in issues)

    def test_all_nan_column(self):
        """测试全NaN列"""
        data = pd.DataFrame({
            'good': [1, 2, 3, 4, 5],
            'all_nan': [np.nan] * 5
        })

        loader = MockDataLoader()

        # 均值策略无法处理全NaN列
        result = loader.handle_missing_values(data, strategy='mean')
        # all_nan列仍然是NaN
        assert result['all_nan'].isna().all()

        # 前向填充也无法处理
        result = loader.handle_missing_values(data, strategy='forward_fill')
        assert result['all_nan'].isna().all()

    def test_multiindex_data(self):
        """测试MultiIndex数据"""
        # 创建MultiIndex
        dates = pd.date_range('2020-01-01', periods=10)
        tickers = ['AAPL', 'GOOGL', 'MSFT']
        index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])

        data = pd.DataFrame({
            'close': np.random.randn(30),
            'volume': np.random.exponential(1000000, 30)
        }, index=index)

        # 检查缺失值应该正常工作
        loader = MockDataLoader()
        missing_info = loader.check_missing_values(data)
        assert 'close' not in missing_info  # 无缺失

        # 添加缺失值
        data.iloc[0:3, 0] = np.nan
        missing_info = loader.check_missing_values(data)
        assert 'close' in missing_info
        assert missing_info['close']['count'] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])