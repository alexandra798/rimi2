import pandas as pd, numpy as np
from alpha.evaluator import FormulaEvaluator

X = pd.DataFrame({
    'close': pd.Series([np.nan]*10 + list(np.random.randn(90))),
    'volume': pd.Series([np.nan]*5 + list(np.random.rand(95)))
})
ev = FormulaEvaluator()

# 这些公式在补丁前容易触发 All-NaN slice warning；补丁后应返回有限数
for f in [
    'BEG close ts_med delta_20 END',
    'BEG close volume corr delta_10 END',
    'BEG close ts_ref delta_5 ts_std delta_5 END'
]:
    s = ev.evaluate(f, X, allow_partial=False)
    assert s.notna().sum() > 0
    print(f, "ok")

# 没有问题