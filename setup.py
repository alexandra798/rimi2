from setuptools import setup, find_packages

setup(
    name="riskminer_algorithm",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.4",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "optuna>=4.0.0",
        "torch>=2.0.0",
    ],
    python_requires=">=3.8",
)