import numpy as np
from coolpath.analysis.statistics import fdr_bh, robust_correlation


def test_fdr():
    reject, q = fdr_bh([.001, .01, .2, .8], alpha=.05)
    assert reject.tolist() == [True, True, False, False]
    assert np.all((q[np.isfinite(q)] >= 0) & (q[np.isfinite(q)] <= 1))


def test_correlation():
    x = np.arange(20.)
    r = robust_correlation(x, 2 * x + 1)
    assert r['r'] > .999 and r['rho'] > .999
