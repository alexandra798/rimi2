#!/usr/bin/env python
"""运行所有测试的脚本"""

import sys
import os
import pytest
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_tests(test_dir="tests", verbose=False, coverage=False, specific_test=None):
    """
    运行测试套件

    Args:
        test_dir: 测试文件目录
        verbose: 是否显示详细输出
        coverage: 是否生成覆盖率报告
        specific_test: 运行特定的测试文件或测试函数

    Returns:
        退出码（0表示成功）
    """
    # 构建pytest参数
    args = []

    # 添加测试目录或特定测试
    if specific_test:
        args.append(specific_test)
    else:
        args.append(test_dir)

    # 详细输出
    if verbose:
        args.append("-v")
        args.append("-s")  # 显示print输出

    # 覆盖率
    if coverage:
        args.extend([
            "--cov=.",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-config=.coveragerc"
        ])

    # 其他有用的选项
    args.extend([
        "--tb=short",  # 简短的traceback
        "--strict-markers",  # 严格的标记
        "-p", "no:warnings",  # 禁用警告（可选）
    ])

    # 运行pytest
    print(f"Running tests with args: {' '.join(args)}")
    return pytest.main(args)


def run_specific_module(module_name):
    """运行特定模块的测试"""
    module_tests = {
        "token": "tests/test_token_system.py",
        "rpn": "tests/test_rpn_evaluator.py",
        "alpha": "tests/test_alpha_pool.py",
        "mcts": "tests/test_mcts.py",
        "policy": "tests/test_policy_network.py",
        "data": "tests/test_data_loader.py",
    }

    if module_name in module_tests:
        return run_tests(specific_test=module_tests[module_name], verbose=True)
    else:
        print(f"Unknown module: {module_name}")
        print(f"Available modules: {', '.join(module_tests.keys())}")
        return 1


def run_quick_tests():
    """运行快速测试（排除慢测试）"""
    args = [
        "tests",
        "-v",
        "-m", "not slow",  # 排除标记为slow的测试
        "--tb=short",
    ]
    return pytest.main(args)


def run_integration_tests():
    """只运行集成测试"""
    args = [
        "tests",
        "-v",
        "-k", "Integration",  # 只运行名称包含Integration的测试
        "--tb=short",
    ]
    return pytest.main(args)


def check_dependencies():
    """检查测试依赖"""
    required_packages = [
        "pytest",
        "pytest-cov",
        "numpy",
        "pandas",
        "torch",
        "scipy",
        "sklearn"
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)

    if missing:
        print("Missing tests dependencies:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with: pip install " + " ".join(missing))
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Run tests suite for RiskMiner project")

    parser.add_argument(
        "--module", "-m",
        help="Run tests for specific module (token/rpn/alpha/mcts/policy/data)"
    )
    parser.add_argument(
        "--tests", "-t",
        help="Run specific tests file or tests function"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run quick tests only (exclude slow tests)"
    )
    parser.add_argument(
        "--integration", "-i",
        action="store_true",
        help="Run integration tests only"
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check tests dependencies"
    )

    args = parser.parse_args()

    # 检查依赖
    if args.check_deps:
        if check_dependencies():
            print("All tests dependencies are installed ✓")
            return 0
        else:
            return 1

    # 确保测试目录存在
    test_dir = Path("tests")
    if not test_dir.exists():
        print(f"Creating tests directory: {test_dir}")
        test_dir.mkdir(exist_ok=True)

    # 运行相应的测试
    if args.module:
        exit_code = run_specific_module(args.module)
    elif args.tests:
        exit_code = run_tests(specific_test=args.test, verbose=True)
    elif args.quick:
        exit_code = run_quick_tests()
    elif args.integration:
        exit_code = run_integration_tests()
    else:
        exit_code = run_tests(
            verbose=args.verbose,
            coverage=args.coverage
        )

    # 打印结果
    if exit_code == 0:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ Tests failed with exit code: {exit_code}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())