from __future__ import annotations
import numpy as np
from scipy import stats


def autocorr_lag1(v) -> float:
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    v = v[ok]
    if len(v) < 3 or np.std(v[:-1]) == 0 or np.std(v[1:]) == 0: return 0.0
    return float(np.corrcoef(v[:-1], v[1:])[0, 1])


def robust_correlation(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {
            'N': n,
            'N_eff': np.nan,
            'r': np.nan,
            'rho': np.nan,
            'p_pearson': np.nan,
            'p_spearman': np.nan,
            'p_corrigee': np.nan
        }
    pear = stats.pearsonr(x, y)
    spear = stats.spearmanr(x, y)
    rx, ry = autocorr_lag1(x), autocorr_lag1(y)
    neff = float(np.clip(n * (1 - rx * ry) / max(1 + rx * ry, 1e-12), 3, n))
    t = pear.statistic * np.sqrt(
        (neff - 2) / max(1 - pear.statistic**2, 1e-12))
    p_corr = float(2 * stats.t.sf(abs(t), neff - 2))
    return {
        'N': n,
        'N_eff': neff,
        'r': float(pear.statistic),
        'rho': float(spear.statistic),
        'p_pearson': float(pear.pvalue),
        'p_spearman': float(spear.pvalue),
        'p_corrigee': p_corr
    }


def fdr_bh(pvals, alpha=.05):
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    idx = np.where(ok)[0]
    pv = p[ok]
    n = len(pv)
    reject = np.zeros(len(p), bool)
    q = np.full(len(p), np.nan)
    if n == 0: return reject, q
    order = np.argsort(pv)
    ps = pv[order]
    passes = ps <= alpha * np.arange(1, n + 1) / n
    r = np.zeros(n, bool)
    if passes.any(): r[:np.max(np.where(passes)[0]) + 1] = True
    qs = np.minimum.accumulate((ps * n / np.arange(1, n + 1))[::-1])[::-1]
    reject[idx[order]] = r
    q[idx[order]] = np.minimum(qs, 1.0)
    return reject, q


def partial_spearman_block_permutation(x,
                                       y,
                                       controls,
                                       n_perm=999,
                                       block_size=None,
                                       seed=42):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    C = [np.asarray(c, float) for c in controls]
    ok = np.isfinite(x) & np.isfinite(y)
    for c in C:
        ok &= np.isfinite(c)
    x, y = x[ok], y[ok]
    C = [c[ok] for c in C]
    n = len(x)
    if n < 5 + len(C): return np.nan, np.nan, n
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    Z = np.column_stack([np.ones(n)] + [stats.rankdata(c) for c in C])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    if np.std(ex) == 0 or np.std(ey) == 0: return np.nan, np.nan, n
    obs = float(np.corrcoef(ex, ey)[0, 1])
    block_size = block_size or 3
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    extreme = 0
    for _ in range(n_perm):
        order = rng.permutation(n_blocks)
        ids = np.concatenate([
            np.arange(b * block_size, min((b + 1) * block_size, n))
            for b in order
        ])[:n]
        r = float(np.corrcoef(ex[ids], ey)[0, 1])
        extreme += int(np.isfinite(r) and abs(r) >= abs(obs))
    return obs, (extreme + 1) / (n_perm + 1), n


def morans_i(values, t, bandwidth=120.0, n_perm=999, seed=42):
    x = np.asarray(values, float)
    t = np.asarray(t, float)
    ok = np.isfinite(x) & np.isfinite(t)
    x, t = x[ok], t[ok]
    n = len(x)
    if n < 4: return np.nan, np.nan, n
    d = np.abs(t[:, None] - t[None, :])
    W = np.exp(-(d / bandwidth)**2)
    np.fill_diagonal(W, 0)
    s0 = W.sum()
    z = x - x.mean()
    den = (z * z).sum()
    obs = float(n / s0 * (W * (z[:, None] * z[None, :])).sum() / den)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        zp = rng.permutation(z)
        val = n / s0 * (W * (zp[:, None] * zp[None, :])).sum() / den
        extreme += int(abs(val) >= abs(obs))
    return obs, (extreme + 1) / (n_perm + 1), n


def ols(y, X, names):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[ok], X[ok]
    D = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(D, y, rcond=None)[0]
    resid = y - D @ beta
    dof = max(len(y) - D.shape[1], 1)
    s2 = float(resid @ resid / dof)
    cov = s2 * np.linalg.pinv(D.T @ D)
    se = np.sqrt(np.diag(cov))
    tstat = beta / se
    p = 2 * stats.t.sf(np.abs(tstat), dof)
    return names and {
        'names': ['const'] + list(names),
        'beta': beta,
        'se': se,
        'p': p,
        'n': len(y),
        'resid': resid,
        'design': D
    }


def vif(X, names):
    X = np.asarray(X, float)
    rows = []
    for j, name in enumerate(names):
        y = X[:, j]
        Z = np.delete(X, j, axis=1)
        ok = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
        yy = y[ok]
        ZZ = Z[ok]
        if len(yy) < 3 or np.std(yy) == 0: val = np.nan
        else:
            D = np.column_stack([np.ones(len(yy)), ZZ])
            pred = D @ np.linalg.lstsq(D, yy, rcond=None)[0]
            ssr = ((yy - pred)**2).sum()
            sst = ((yy - yy.mean())**2).sum()
            r2 = 1 - ssr / sst if sst else np.nan
            val = 1 / max(1 - r2, 1e-12) if np.isfinite(r2) else np.nan
        rows.append({'variable': name, 'VIF': val})
    return rows


def ols_hac(y, X, names, max_lag=None):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[ok], X[ok]
    n = len(y)
    D = np.column_stack([np.ones(n), X])
    beta = np.linalg.lstsq(D, y, rcond=None)[0]
    resid = y - D @ beta
    k = D.shape[1]
    bread = np.linalg.pinv(D.T @ D)
    L = max_lag if max_lag is not None else max(1, int(4 * (n / 100)**(2 / 9)))
    meat = np.zeros((k, k))
    for t in range(n):
        meat += resid[t]**2 * np.outer(D[t], D[t])
    for lag in range(1, min(L, n - 1) + 1):
        weight = 1 - lag / (L + 1)
        gamma = np.zeros((k, k))
        for t in range(lag, n):
            gamma += resid[t] * resid[t - lag] * np.outer(D[t], D[t - lag])
        meat += weight * (gamma + gamma.T)
    cov = bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    z = beta / se
    p = 2 * stats.norm.sf(np.abs(z))
    return {
        'names': ['const'] + list(names),
        'beta': beta,
        'se_hac': se,
        'p_hac': p,
        'n': n,
        'lag': L
    }
