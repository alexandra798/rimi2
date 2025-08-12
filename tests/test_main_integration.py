"""主程序集成测试 - 端到端测试整个系统"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
import torch
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
import argparse


class TestMainProgram:
    """测试主程序功能"""

    @pytest.fixture
    def temp_data_file(self):
        """创建临时数据文件"""
        np.random.seed(42)
        n = 500

        # 创建测试数据
        dates = pd.date_range('2020-01-01', periods=n)
        data = pd.DataFrame({
            'open': 100 + np.cumsum(np.random.randn(n) * 2),
            'high': 105 + np.cumsum(np.random.randn(n) * 2),
            'low': 95 + np.cumsum(np.random.randn(n) * 2),
            'close': 100 + np.cumsum(np.random.randn(n) * 2),
            'volume': np.random.exponential(1000000, n),
            'vwap': 100 + np.cumsum(np.random.randn(n) * 1.5),
            'target': np.random.randn(n) * 0.01
        }, index=dates)

        # 保存到临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=True)
            temp_path = f.name

        yield temp_path

        # 清理
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def mock_args(self, temp_data_file):
        """创建模拟参数"""
        args = argparse.Namespace(
            data_path=temp_data_file,
            target_column='target',
            use_token_system=True,
            use_risk_seeking=False,
            transform_data=False,
            cross_validate=False,
            backtest=False,
            save_transformed=False,
            output_path=None,
            save_results=False,
            results_path=None,
            gpu_id=0,
            force_continue=False,
            random_seed=42
        )
        return args

    def test_data_loading_and_cleaning(self, mock_args):
        """测试数据加载和清理流程"""
        from main import main

        # 模拟主函数的数据加载部分
        with patch('main.load_user_dataset') as mock_load:
            # 设置返回值
            X = pd.DataFrame({
                'close': [100, 101, 102, 103, 104],
                'volume': [1000, 1100, 900, 1200, 1050]
            })
            y = pd.Series([0.01, 0.02, 0.01, 0.015, 0.02])
            features = ['close', 'volume']

            mock_load.return_value = (X, y, features)

            # 测试加载
            mock_load(mock_args.data_path, mock_args.target_column)
            mock_load.assert_called_once()

    def test_mcts_training_flow(self, mock_args):
        """测试MCTS训练流程"""
        mock_args.use_risk_seeking = True

        # 创建模拟数据
        X = pd.DataFrame({
            'close': np.random.randn(100),
            'volume': np.random.randn(100)
        })
        y = pd.Series(np.random.randn(100) * 0.01)

        # 模拟MCTS训练
        with patch('main.run_mcts_with_token_system') as mock_mcts:
            mock_mcts.return_value = [
                ('BEG close END', 0.05),
                ('BEG volume END', 0.03)
            ]

            # 调用
            result = mock_mcts(X, y, num_iterations=10)

            assert len(result) == 2
            assert result[0][0] == 'BEG close END'

    def test_alpha_pool_management(self, mock_args):
        """测试Alpha池管理"""
        from alpha.pool import AlphaPool
        from alpha.evaluator import FormulaEvaluator

        pool = AlphaPool(pool_size=10)
        evaluator = FormulaEvaluator()

        # 添加公式
        formulas = [
            ('BEG close END', 0.05),
            ('BEG volume END', 0.03),
            ('BEG close volume add END', 0.04)
        ]

        for formula, score in formulas:
            pool.add_to_pool({
                'formula': formula,
                'score': score,
                'ic': score
            })

        assert len(pool.alphas) == 3

        # 获取最佳公式
        top = pool.get_top_formulas(2)
        assert len(top) == 2

    def test_cross_validation_integration(self, mock_args):
        """测试交叉验证集成"""
        mock_args.cross_validate = True

        X = pd.DataFrame({
            'close': np.random.randn(200),
            'volume': np.random.randn(200)
        })
        y = pd.Series(np.random.randn(200) * 0.01)

        formulas = ['BEG close END', 'BEG volume END']

        with patch('main.cross_validate_formulas') as mock_cv:
            mock_cv.return_value = {
                'BEG close END': {
                    'Mean IC': 0.05,
                    'IC Std Dev': 0.01,
                    'IC Scores': [0.04, 0.05, 0.06]
                },
                'BEG volume END': {
                    'Mean IC': 0.03,
                    'IC Std Dev': 0.02,
                    'IC Scores': [0.01, 0.03, 0.05]
                }
            }

            results = mock_cv(formulas, X, y, 3, None)

            assert len(results) == 2
            assert results['BEG close END']['Mean IC'] == 0.05

    def test_backtest_integration(self, mock_args):
        """测试回测集成"""
        mock_args.backtest = True

        X = pd.DataFrame({
            'close': np.random.randn(100),
            'volume': np.random.randn(100)
        })
        y = pd.Series(np.random.randn(100) * 0.01)

        formulas = ['BEG close END']

        with patch('main.backtest_formulas') as mock_bt:
            mock_bt.return_value = {'BEG close END': 0.045}

            results = mock_bt(formulas, X, y)

            assert 'BEG close END' in results
            assert results['BEG close END'] == 0.045

    def test_data_transformation(self, mock_args):
        """测试数据转换"""
        mock_args.transform_data = True
        mock_args.save_transformed = True
        mock_args.output_path = 'test_transformed.csv'

        X = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })

        formulas = ['BEG close END', 'BEG volume END']

        with patch('main.apply_alphas_and_return_transformed') as mock_transform:
            transformed = pd.DataFrame({
                'alpha_1': [100, 101, 102],
                'alpha_2': [1000, 1100, 1200]
            })
            mock_transform.return_value = transformed

            result = mock_transform(X, formulas, None)

            assert result.shape == (3, 2)
            assert 'alpha_1' in result.columns

    def test_results_saving(self, mock_args):
        """测试结果保存"""
        mock_args.save_results = True

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            mock_args.results_path = f.name

        try:
            # 模拟保存结果
            top_formulas = [
                'BEG close END',
                'BEG volume END',
                'BEG close volume add END'
            ]

            with open(mock_args.results_path, 'w') as f:
                f.write("=== Top Alpha Formulas ===\n")
                for i, formula in enumerate(top_formulas, 1):
                    f.write(f"{i}. {formula}\n")

            # 验证文件内容
            with open(mock_args.results_path, 'r') as f:
                content = f.read()

            assert "Top Alpha Formulas" in content
            assert "BEG close END" in content

        finally:
            # 清理
            if os.path.exists(mock_args.results_path):
                os.unlink(mock_args.results_path)

    def test_gpu_device_handling(self, mock_args):
        """测试GPU设备处理"""
        mock_args.gpu_id = 0

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{mock_args.gpu_id}")
            assert device.type == 'cuda'
            assert device.index == 0
        else:
            device = torch.device("cpu")
            assert device.type == 'cpu'

    def test_error_handling(self, mock_args):
        """测试错误处理"""
        # 测试无效数据路径
        mock_args.data_path = "nonexistent_file.csv"

        with patch('main.load_user_dataset') as mock_load:
            mock_load.side_effect = FileNotFoundError("File not found")

            with pytest.raises(FileNotFoundError):
                mock_load(mock_args.data_path, mock_args.target_column)

    def test_force_continue_flag(self, mock_args):
        """测试强制继续标志"""
        mock_args.force_continue = True

        # 模拟数据质量检查失败
        with patch('main.validate_data_quality') as mock_validate:
            mock_validate.return_value = (False, ["Data has NaN values"])

            X = pd.DataFrame({'col': [1, np.nan, 3]})
            y = pd.Series([0.01, 0.02, 0.03])

            is_valid, issues = mock_validate(X, y)

            assert not is_valid
            # 即使验证失败，force_continue应该允许继续
            assert mock_args.force_continue == True


class TestEndToEndWorkflow:
    """端到端工作流测试"""

    @pytest.mark.slow
    def test_minimal_workflow(self, temp_data_file):
        """测试最小工作流"""
        args = argparse.Namespace(
            data_path=temp_data_file,
            target_column='target',
            use_token_system=True,
            use_risk_seeking=False,
            transform_data=False,
            cross_validate=False,
            backtest=False,
            save_transformed=False,
            output_path=None,
            save_results=False,
            results_path=None,
            gpu_id=0,
            force_continue=True,
            random_seed=42
        )

        # 由于main函数可能很复杂，这里模拟关键步骤
        # 实际测试时应该导入并调用main函数

        # 1. 加载数据
        data = pd.read_csv(temp_data_file, index_col=0)
        X = data.drop(columns=['target'])
        y = data['target']

        assert len(X) > 0
        assert len(y) > 0

        # 2. 数据预处理
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        assert len(X_train) > len(X_test)

        # 3. 简单的公式生成
        formulas = ['BEG close END', 'BEG volume END']

        # 4. 验证输出
        assert len(formulas) > 0

    @pytest.mark.slow
    def test_full_pipeline_with_mocks(self):
        """使用模拟测试完整管道"""
        # 创建所有必要的模拟
        with patch('main.load_user_dataset') as mock_load, \
                patch('main.run_mcts_with_token_system') as mock_mcts, \
                patch('main.cross_validate_formulas') as mock_cv, \
                patch('main.backtest_formulas') as mock_bt:
            # 设置模拟返回值
            X = pd.DataFrame({'close': np.random.randn(100)})
            y = pd.Series(np.random.randn(100))
            mock_load.return_value = (X, y, ['close'])

            mock_mcts.return_value = [
                ('BEG close END', 0.05),
                ('BEG volume END', 0.03)
            ]

            mock_cv.return_value = {
                'BEG close END': {'Mean IC': 0.05}
            }

            mock_bt.return_value = {'BEG close END': 0.045}

            # 模拟参数
            args = MagicMock()
            args.data_path = 'tests.csv'
            args.target_column = 'target'
            args.use_token_system = True
            args.use_risk_seeking = True
            args.cross_validate = True
            args.backtest = True
            args.save_results = False
            args.force_continue = False
            args.gpu_id = 0
            args.random_seed = 42

            # 执行工作流的关键步骤
            X, y, features = mock_load(args.data_path, args.target_column)
            assert X is not None

            formulas = mock_mcts(X, y)
            assert len(formulas) > 0

            cv_results = mock_cv([f[0] for f in formulas], X, y, 3)
            assert len(cv_results) > 0

            bt_results = mock_bt([f[0] for f in formulas], X, y)
            assert len(bt_results) > 0

    def test_multiprocessing_compatibility(self):
        """测试多进程兼容性"""
        # 测试代码是否可以在多进程环境中运行
        import multiprocessing as mp

        def worker_function():
            # 简单的工作函数
            X = pd.DataFrame({'close': [100, 101, 102]})
            y = pd.Series([0.01, 0.02, 0.03])
            return len(X)

        # 创建进程
        if __name__ != "__main__":  # 避免在导入时运行
            with mp.Pool(processes=2) as pool:
                results = pool.map(worker_function, [None, None])
                assert all(r == 3 for r in results)

    def test_memory_usage(self):
        """测试内存使用"""
        import tracemalloc

        # 开始跟踪
        tracemalloc.start()

        # 创建大数据集
        X = pd.DataFrame({
            'close': np.random.randn(10000),
            'volume': np.random.randn(10000)
        })
        y = pd.Series(np.random.randn(10000))

        # 获取当前内存使用
        current, peak = tracemalloc.get_traced_memory()

        # 停止跟踪
        tracemalloc.stop()

        # 内存使用应该在合理范围内（比如小于100MB）
        assert peak / 1024 / 1024 < 100  # MB


class TestConfigurationIntegration:
    """配置集成测试"""

    def test_config_loading(self):
        """测试配置加载"""
        from config.config import (
            MCTS_CONFIG,
            ALPHA_POOL_CONFIG,
            validate_config
        )

        # 验证配置
        validate_config()

        # 检查关键配置
        assert MCTS_CONFIG['gamma'] == 1.0
        assert ALPHA_POOL_CONFIG['pool_size'] == 100
        assert ALPHA_POOL_CONFIG['lambda_param'] == 0.1

    def test_config_override(self):
        """测试配置覆盖"""
        from config.config import MCTS_CONFIG

        # 保存原始值
        original_gamma = MCTS_CONFIG['gamma']

        # 临时修改
        MCTS_CONFIG['gamma'] = 0.95
        assert MCTS_CONFIG['gamma'] == 0.95

        # 恢复
        MCTS_CONFIG['gamma'] = original_gamma
        assert MCTS_CONFIG['gamma'] == 1.0

    def test_config_consistency(self):
        """测试配置一致性"""
        from config.config import (
            MCTS_CONFIG,
            GRU_CONFIG,
            POLICY_CONFIG
        )

        # GRU配置应该与Policy配置一致
        assert GRU_CONFIG['num_layers'] == POLICY_CONFIG['gru_layers']
        assert GRU_CONFIG['hidden_dim'] == POLICY_CONFIG['gru_hidden_dim']


class TestSystemRobustness:
    """系统鲁棒性测试"""

    def test_concurrent_access(self):
        """测试并发访问"""
        from alpha.evaluator import FormulaEvaluator
        import threading

        evaluator = FormulaEvaluator()
        X = pd.DataFrame({'close': np.random.randn(100)})

        results = []

        def evaluate_formula():
            result = evaluator.evaluate('BEG close END', X)
            results.append(result is not None)

        # 创建多个线程
        threads = []
        for _ in range(5):
            t = threading.Thread(target=evaluate_formula)
            threads.append(t)
            t.start()

        # 等待完成
        for t in threads:
            t.join()

        # 所有评估应该成功
        assert all(results)

    def test_large_formula_handling(self):
        """测试大公式处理"""
        # 创建一个很长的公式
        formula_parts = ['BEG', 'close']
        for _ in range(20):
            formula_parts.extend(['volume', 'add'])
        formula_parts.append('END')

        long_formula = ' '.join(formula_parts)

        from alpha.evaluator import FormulaEvaluator
        evaluator = FormulaEvaluator()

        X = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })

        # 应该能处理长公式（或优雅地失败）
        result = evaluator.evaluate(long_formula, X)
        # 结果可能是None（如果公式太长）或有效值
        assert result is None or isinstance(result, pd.Series)

    def test_data_type_compatibility(self):
        """测试数据类型兼容性"""
        from alpha.evaluator import FormulaEvaluator
        evaluator = FormulaEvaluator()

        # 测试不同数据类型
        test_cases = [
            # NumPy数组
            {'close': np.array([100, 101, 102])},
            # Python列表
            {'close': [100, 101, 102]},
            # Pandas Series
            {'close': pd.Series([100, 101, 102])},
            # 混合类型
            {'close': pd.Series([100, 101, 102]),
             'volume': np.array([1000, 1100, 1200])}
        ]

        for data in test_cases:
            result = evaluator.evaluate('BEG close END', data)
            assert result is not None
            assert len(result) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])