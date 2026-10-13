#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reproduce.py -- Replication script for
"Screening Green Shippers under Fare Regulation: Menu Pricing and Subsidy
Incidence in Urban Rail Freight"

Usage
-----
    python3 reproduce.py         core numbers + Figures 1, 2, 3, 5 (about a minute)
    python3 reproduce.py --mc    additionally reruns the Monte Carlo study
                                 (9 designs x N = 2000 draws; a few minutes),
                                 caches the draws to figs/mc_cache.npz and draws
                                 Figures 4 and 6. Once the cache exists, plain
                                 `python3 reproduce.py` redraws all six figures.

Requirements: Python >= 3.10, NumPy, SciPy, Matplotlib. Nothing else.

Reproducibility contract (matches the Data Availability statement):
  * single deterministic script; Monte Carlo uses numpy.random.default_rng(42)
    with common random numbers across designs;
  * draw order per replication: alpha, beta, T, c_N, eta, then the cost gap c_B - c_N;
  * units: money in 10^4 CNY/yr, quantities in 10^4 t/yr, prices/subsidies in CNY per ton.

The script prints every headline number of the paper together with the value
reported in the text and an OK/check flag (tolerance 0.5% unless stated).
"""
import os
import sys
import numpy as np
from scipy.optimize import brentq, minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, 'figs')
os.makedirs(FIGDIR, exist_ok=True)
CACHE = os.path.join(FIGDIR, 'mc_cache.npz')

# ---------------------------------------------------------------- parameters
P0 = dict(rho=0.30, kappa=0.15, mu=100.0, alpha=800.0, beta=1.5, gamma=0.0,
          pH=400.0, cN=360.0, cB=380.0, F=150.0, pbar=450.0, omega=1.2,
          pc=60.0, egap=0.084, d=30.0, phi=0.05, eta=8.0, T=15.0, K=0.101)

# ------------------------------------------------------------ model primitives
def margins(P):
    """A_B, A_N (demand intercepts) and maximal operating margins M_B, M_N."""
    AB = P['alpha'] + P['beta']*P['T'] + P['gamma']*P['pH']
    AN = P['alpha'] + P['gamma']*P['pH']
    return AB, AN, AB/P['beta'] - P['cB'], AN/P['beta'] - P['cN']

def qB(P, pB):
    AB, _, _, _ = margins(P)
    return P['rho']*(AB - P['beta']*pB)

def qN(P, pN):
    _, AN, _, _ = margins(P)
    return (1 - P['rho'])*(AN - P['beta']*pN)

def operator(P, sB, sN):
    """Operator best response by active-set enumeration over the five KKT regimes."""
    AB, AN, MB, MN = margins(P)
    b, pb, T = P['beta'], P['pbar'], P['T']
    r = P['rho']
    pBu = (AB + b*(P['cB'] - sB))/(2*b)
    pNu = (AN + b*(P['cN'] - sN))/(2*b)
    def prof(pB, pN):
        return (pB + sB - P['cB'])*qB(P, pB) + (pN + sN - P['cN'])*qN(P, pN) - P['F']
    cands = []
    # regime I: interior
    if 0 <= pBu - pNu <= T and pBu <= pb and pNu <= pb:
        cands.append((pBu, pNu, 'I'))
    # regime IIa: upper IC binds (pB = pN + T)
    pN2 = (r*AB + (1-r)*AN - r*b*(sB - P['cB']) - (1-r)*b*(sN - P['cN']) - 2*r*b*T)/(2*b)
    if pN2 + T <= pb and pN2 >= 0:
        cands.append((pN2 + T, pN2, 'IIa'))
    # regime IIb: uniform price (both ICs bind)
    pU = (r*AB + (1-r)*AN + r*b*(P['cB'] - sB) + (1-r)*b*(P['cN'] - sN))/(2*b)
    if 0 <= pU <= pb:
        cands.append((pU, pU, 'IIb'))
    # regimes III/IV: cap binds on the green service
    if pBu > pb:
        pN3 = min(max(pNu, pb - T), pb)
        reg = 'IV' if abs(pN3 - (pb - T)) < 1e-9 else 'III'
        cands.append((pb, pN3, reg))
        cands.append((pb, pb, 'III'))
    if pNu > pb:
        cands.append((pb, pb, 'III'))
    best_ = None
    for pB_, pN_, reg in cands:
        if pB_ < -1e-9 or pN_ < -1e-9 or pB_ > pb + 1e-9 or pN_ > pb + 1e-9:
            continue
        if not (-1e-9 <= pB_ - pN_ <= T + 1e-9):
            continue
        pi = prof(pB_, pN_)
        if best_ is None or pi > best_[2] + 1e-12:
            best_ = (pB_, pN_, pi, reg)
    pB_, pN_, pi, reg = best_
    return dict(pB=pB_, pN=pN_, profit=pi, regime=reg, qB=qB(P, pB_), qN=qN(P, pN_))

def fallback_profit(P, sN):
    """Pi_N^0(s_N): pooled traditional-only profit at the operator-optimal price."""
    _, AN, _, MN = margins(P)
    b, pb = P['beta'], P['pbar']
    pu = (AN + b*(P['cN'] - sN))/(2*b)
    p = min(pu, pb)
    return (p + sN - P['cN'])*(AN - b*p)

def active(P, sB, sN, op=None):
    if op is None:
        op = operator(P, sB, sN)
    return op['profit'] >= fallback_profit(P, sN) - 1e-9

def welfare(P, sB, sN, op=None):
    """W = CS + profit + K*(qB+qN) + eta*qB - omega*(sB*qB + sN*qN)."""
    if op is None:
        op = operator(P, sB, sN)
    b, r = P['beta'], P['rho']
    CS = op['qB']**2/(2*r*b) + op['qN']**2/(2*(1-r)*b)
    return (CS + op['profit'] + P['K']*(op['qB'] + op['qN']) + P['eta']*op['qB']
            - P['omega']*(sB*op['qB'] + sN*op['qN']))

def scrit(P):
    """s_B^crit = c_B + A_B/beta - 2*pbar: release threshold of the locked region."""
    AB, _, _, _ = margins(P)
    return P['cB'] + AB/P['beta'] - 2*P['pbar']

def sBmin_locked(P):
    """s_B^min: smallest activating green subsidy in the locked region."""
    f = lambda sB: operator(P, sB, 0.0)['profit'] - fallback_profit(P, 0.0)
    if f(0.0) >= 0:
        return 0.0
    return brentq(f, 0.0, 500.0, xtol=1e-10)

# ------------------------------------------------------- government programs
def best_separating(P):
    """Best separating menu: regime-I FOC, activation boundary, locked corner,
    green-only path (fast closed-form candidate list)."""
    AB, AN, MB, MN = margins(P)
    b, r, pb, T = P['beta'], P['rho'], P['pbar'], P['T']
    om, K, eta, F = P['omega'], P['K'], P['eta'], P['F']
    cands = []
    sBh = ((3 - 2*om)*MB + 2*(K + eta))/(4*om - 3)
    sNh = ((3 - 2*om)*MN + 2*K)/(4*om - 3)
    cands.append((max(sBh, 0.0), max(sNh, 0.0), 'FOC'))
    fb = lambda sN: np.sqrt(max((MN + sN)**2 + 4*F/(r*b), 0.0)) - MB
    def Wb(sN):
        sB = fb(sN)
        op = operator(P, sB, sN)
        if op['regime'] != 'I':
            return -1e18
        return welfare(P, sB, sN, op)
    res = minimize_scalar(lambda x: -Wb(x), bounds=(0.0, 400.0), method='bounded',
                          options={'xatol': 1e-10})
    cands.append((fb(res.x), res.x, 'boundary'))
    cands.append((fb(0.0), 0.0, 'boundary0'))
    cands.append((sBmin_locked(P), 0.0, 'locked'))
    def Wg(sB):
        op = operator(P, sB, 0.0)
        if op['profit'] < fallback_profit(P, 0.0) - 1e-9:
            return -1e18
        return welfare(P, sB, 0.0, op)
    grid = np.linspace(0, 400, 801)
    vals = np.array([Wg(s) for s in grid])
    i = int(np.argmax(vals))
    lo, hi = grid[max(i-1, 0)], grid[min(i+1, len(grid)-1)]
    res2 = minimize_scalar(lambda x: -Wg(x), bounds=(lo, hi), method='bounded',
                           options={'xatol': 1e-10})
    cands.append((res2.x, 0.0, 'greenonly'))
    out = None
    for sB, sN, tag in cands:
        op = operator(P, sB, sN)
        if op['profit'] < fallback_profit(P, sN) - 1e-9:
            continue
        W = welfare(P, sB, sN, op)
        if out is None or W > out['W'] + 1e-12:
            out = dict(sB=sB, sN=sN, W=W, tag=tag, op=op)
    return out

def operator_grid(P, sBm, sNm):
    """Vectorized operator best response over meshgrid arrays (same shape)."""
    AB, AN, MB, MN = margins(P)
    b, pb, T, r, F = P['beta'], P['pbar'], P['T'], P['rho'], P['F']
    cB, cN = P['cB'], P['cN']
    pBu = (AB + b*(cB - sBm))/(2*b)
    pNu = (AN + b*(cN - sNm))/(2*b)
    def prof(pB, pN):
        return ((pB + sBm - cB)*r*np.maximum(AB - b*pB, 0)
                + (pN + sNm - cN)*(1-r)*np.maximum(AN - b*pN, 0) - F)
    best = np.full(sBm.shape, -1e18)
    bpB = np.zeros_like(best); bpN = np.zeros_like(best)
    def upd(pB, pN, mask):
        nonlocal best, bpB, bpN
        ok = (mask & (pB >= -1e-9) & (pN >= -1e-9) & (pB <= pb + 1e-9) & (pN <= pb + 1e-9)
              & (pB - pN >= -1e-9) & (pB - pN <= T + 1e-9))
        v = np.where(ok, prof(np.clip(pB, 0, pb), np.clip(pN, 0, pb)), -1e18)
        imp = v > best
        best = np.where(imp, v, best)
        bpB = np.where(imp, pB, bpB); bpN = np.where(imp, pN, bpN)
    upd(pBu, pNu, np.ones_like(best, bool))                                     # I
    pN2 = (r*AB + (1-r)*AN - r*b*(sBm - cB) - (1-r)*b*(sNm - cN) - 2*r*b*T)/(2*b)
    upd(pN2 + T, pN2, np.ones_like(best, bool))                                 # IIa
    pU = (r*AB + (1-r)*AN + r*b*(cB - sBm) + (1-r)*b*(cN - sNm))/(2*b)
    upd(pU, pU, np.ones_like(best, bool))                                       # IIb
    upd(np.full_like(best, pb), np.clip(pNu, pb - T, pb), pBu > pb)             # III/IV
    upd(np.full_like(best, pb), np.full_like(best, pb), (pBu > pb) | (pNu > pb))
    return bpB, bpN, best

def best_menu_grid(P, coarse=(81, 61), zoom=2):
    """Grid-based government optimum over the separating menu, with zoom refinement."""
    AB, AN, MB, MN = margins(P)
    b, r, pb = P['beta'], P['rho'], P['pbar']
    om, K, eta, F = P['omega'], P['K'], P['eta'], P['F']
    hiB, hiN = 300.0, 200.0
    best = None
    for z in range(zoom + 1):
        if z == 0:
            sBs = np.linspace(0, hiB, coarse[0]); sNs = np.linspace(0, hiN, coarse[1])
        else:
            sBs = np.linspace(max(best[0] - dB*2, 0), min(best[0] + dB*2, hiB), 41)
            sNs = np.linspace(max(best[1] - dN*2, 0), min(best[1] + dN*2, hiN), 41)
        SBm, SNm = np.meshgrid(sBs, sNs)
        pB, pN, pi = operator_grid(P, SBm, SNm)
        pfb = np.minimum((AN + b*(P['cN'] - SNm))/(2*b), pb)
        pi0 = (pfb + SNm - P['cN'])*np.maximum(AN - b*pfb, 0)
        qBg = r*np.maximum(AB - b*pB, 0); qNg = (1-r)*np.maximum(AN - b*pN, 0)
        CS = qBg**2/(2*r*b) + qNg**2/(2*(1-r)*b)
        W = CS + pi + K*(qBg + qNg) + eta*qBg - om*(SBm*qBg + SNm*qNg)
        W = np.where((pi >= pi0 - 1e-9) & (pi > -1e17), W, -1e18)
        i = np.unravel_index(np.argmax(W), W.shape)
        if W[i] < -1e17:
            return None
        best = (SBm[i], SNm[i], W[i])
        dB = sBs[1] - sBs[0]; dN = sNs[1] - sNs[0]
    return best

def best_menu_fast(P):
    """Closed-form candidate list (regime-I FOC, activation boundary, IIb boundary,
    green-only path)."""
    AB, AN, MB, MN = margins(P)
    b, r, pb, T = P['beta'], P['rho'], P['pbar'], P['T']
    om, K, eta, F = P['omega'], P['K'], P['eta'], P['F']
    cands = []
    sBh = ((3 - 2*om)*MB + 2*(K + eta))/(4*om - 3)
    sNh = ((3 - 2*om)*MN + 2*K)/(4*om - 3)
    cands.append((max(sBh, 0.0), max(sNh, 0.0)))
    cands.append((0.0, 0.0))
    fb = lambda sN: np.sqrt(max((MN + sN)**2 + 4*F/(r*b), 0.0)) - MB
    def Wb(sN):
        sB = fb(sN)
        op = operator(P, sB, sN)
        if op['regime'] != 'I':
            return -1e18
        return welfare(P, sB, sN, op)
    res = minimize_scalar(lambda x: -Wb(x), bounds=(0.0, 400.0), method='bounded',
                          options={'xatol': 1e-8})
    cands.append((fb(res.x), res.x)); cands.append((fb(0.0), 0.0))
    f0 = lambda sB: operator(P, sB, 0.0)['profit'] - fallback_profit(P, 0.0)
    if f0(0.0) >= 0:
        cands.append((0.0, 0.0))
    elif f0(500.0) > 0:
        cands.append((brentq(f0, 0.0, 500.0), 0.0))
    d = (P['cB'] - P['cN']) + T          # regime-I/IIb boundary line sB = sN + d
    def Wl(sN):
        return welfare(P, sN + d, sN, operator(P, sN + d, sN))
    res3 = minimize_scalar(lambda x: -Wl(x), bounds=(0.0, 300.0), method='bounded',
                           options={'xatol': 1e-8})
    cands.append((res3.x + d, res3.x))
    def Wg(sB):
        op = operator(P, sB, 0.0)
        if op['profit'] < fallback_profit(P, 0.0) - 1e-9:
            return -1e18
        return welfare(P, sB, 0.0, op)
    grid = np.linspace(0, 400, 161)
    vals = np.array([Wg(s) for s in grid])
    i = int(np.argmax(vals))
    res2 = minimize_scalar(lambda x: -Wg(x),
                           bounds=(grid[max(i-1, 0)], grid[min(i+1, 160)]),
                           method='bounded', options={'xatol': 1e-8})
    cands.append((res2.x, 0.0))
    out = None
    for sB, sN in cands:
        op = operator(P, sB, sN)
        if op['profit'] < fallback_profit(P, sN) - 1e-9:
            continue
        W = welfare(P, sB, sN, op)
        if out is None or W > out['W'] + 1e-12:
            out = dict(sB=sB, sN=sN, W=W, op=op)
    return out

def best_menu_robust_fast(P):
    """best_menu_fast + coarse grid safety net (41x31; zoom only if the grid wins).
    Guards against local optima of the pooling benchmarks and menu candidates."""
    bm = best_menu_fast(P)
    sBs = np.linspace(0, 300, 41); sNs = np.linspace(0, 200, 31)
    SBm, SNm = np.meshgrid(sBs, sNs)
    AB, AN, MB, MN = margins(P)
    b, r, pb = P['beta'], P['rho'], P['pbar']
    om, K, eta, F = P['omega'], P['K'], P['eta'], P['F']
    pB, pN, pi = operator_grid(P, SBm, SNm)
    pfb = np.minimum((AN + b*(P['cN'] - SNm))/(2*b), pb)
    pi0 = (pfb + SNm - P['cN'])*np.maximum(AN - b*pfb, 0)
    qBg = r*np.maximum(AB - b*pB, 0); qNg = (1-r)*np.maximum(AN - b*pN, 0)
    CS = qBg**2/(2*r*b) + qNg**2/(2*(1-r)*b)
    W = CS + pi + K*(qBg + qNg) + eta*qBg - om*(SBm*qBg + SNm*qNg)
    W = np.where((pi >= pi0 - 1e-9) & (pi > -1e17), W, -1e18)
    i = np.unravel_index(np.argmax(W), W.shape)
    if W[i] > -1e17 and (bm is None or W[i] > bm['W'] + 1e-6):
        g = best_menu_grid(P)
        if g is not None and (bm is None or g[2] > bm['W'] + 1e-6):
            return dict(sB=g[0], sN=g[1], W=g[2], op=operator(P, g[0], g[1]), tag='grid')
    return bm

def poolingN(P):
    """Optimally subsidized pooling on the traditional service (1-D grid + polish).
    The grid is essential: the objective is kinked at the price cap, and a single
    local solve can converge to the interior-FOC local optimum on the wrong side."""
    _, AN, _, MN = margins(P)
    b, pb, om, K = P['beta'], P['pbar'], P['omega'], P['K']
    def W(sN):
        pu = (AN + b*(P['cN'] - sN))/(2*b); p = min(pu, pb)
        q = AN - b*p
        return q**2/(2*b) + (p - P['cN'])*q + K*q - (om - 1)*sN*q
    grid = np.linspace(0, 400, 1601)
    vals = np.array([W(s) for s in grid])
    i = int(np.argmax(vals))
    lo, hi = grid[max(i-1, 0)], grid[min(i+1, 1600)]
    res = minimize_scalar(lambda x: -W(x), bounds=(lo, hi), method='bounded',
                          options={'xatol': 1e-12})
    sN = res.x
    pu = (AN + b*(P['cN'] - sN))/(2*b); p = min(pu, pb); q = AN - b*p
    return dict(sN=sN, p=p, q=q, W=W(sN))

def poolingG(P):
    """Optimally subsidized pooling on an all-green service, subject to the
    operator's participation floor (pooled traditional fallback)."""
    AB, AN, MB, MN = margins(P)
    b, pb, om, K, eta, F = P['beta'], P['pbar'], P['omega'], P['K'], P['eta'], P['F']
    a_eff = P['alpha'] + P['rho']*b*P['T'] + P['gamma']*P['pH']
    p0_ = min((AN + b*P['cN'])/(2*b), pb)
    floor = (p0_ - P['cN'])*(AN - b*p0_)
    def prof(sB):
        pu = (a_eff + b*(P['cB'] - sB))/(2*b); p = min(pu, pb)
        return (p + sB - P['cB'])*(a_eff - b*p) - F
    def W(sB):
        pu = (a_eff + b*(P['cB'] - sB))/(2*b); p = min(pu, pb)
        q = a_eff - b*p
        return q**2/(2*b) + (p - P['cB'])*q + (K + eta)*q - (om - 1)*sB*q - F
    feas_lo = 0.0
    if prof(0.0) < floor:
        if prof(1e6) < floor:
            return dict(sB=np.nan, p=np.nan, q=np.nan, W=-1e18, floor_slack=np.nan)
        feas_lo = brentq(lambda s: prof(s) - floor, 0.0, 1e6)
    hi_search = max(400.0, feas_lo + 50.0)
    grid = np.linspace(feas_lo, hi_search, 1601)
    vals = np.array([W(s) for s in grid])
    i = int(np.argmax(vals))
    lo, hi = grid[max(i-1, 0)], grid[min(i+1, 1600)]
    res = minimize_scalar(lambda x: -W(x), bounds=(lo, hi), method='bounded',
                          options={'xatol': 1e-12})
    sB = res.x
    pu = (a_eff + b*(P['cB'] - sB))/(2*b); p = min(pu, pb); q = a_eff - b*p
    return dict(sB=sB, p=p, q=q, W=W(sB), floor_slack=prof(sB) - floor)

def green_only_mixed(P, eps):
    """Green-only program in regime IIb when a share eps of indifferent P-type
    shipments picks the green service at random (tie-breaking robustness)."""
    b, r, pb = P['beta'], P['rho'], P['pbar']
    AB, AN, MB, MN = margins(P)
    p0_ = min((AN + b*P['cN'])/(2*b), pb)
    floor = (p0_ - P['cN'])*(AN - b*p0_)
    def eq(sB):
        def profit(p):
            qB_ = r*(AB - b*p) + eps*(1-r)*(AN - b*p)
            qN_ = (1-eps)*(1-r)*(AN - b*p)
            return (p + sB - P['cB'])*qB_ + (p - P['cN'])*qN_ - P['F']
        res = minimize_scalar(lambda x: -profit(x), bounds=(200, 600), method='bounded')
        p = res.x
        qB_ = r*(AB - b*p) + eps*(1-r)*(AN - b*p)
        qN_ = (1-eps)*(1-r)*(AN - b*p)
        return p, qB_, qN_, profit(p)
    def W(sB):
        p, qB_, qN_, pi = eq(sB)
        CS = r*(AB - b*p)**2/(2*b) + (1-r)*(AN - b*p)**2/(2*b)
        return CS + pi + P['K']*(qB_ + qN_) + P['eta']*qB_ - P['omega']*sB*qB_
    res = minimize_scalar(lambda x: -W(x), bounds=(0.0, 400.0), method='bounded',
                          options={'xatol': 1e-10})
    sB = res.x
    p, qB_, qN_, pi = eq(sB)
    if pi < floor - 1e-9:
        sB = brentq(lambda s: eq(s)[3] - floor, 0.0, 800.0)
        p, qB_, qN_, pi = eq(sB)
    return dict(sB=sB, p=p, qB=qB_, qN=qN_, W=W(sB))

def with_param(P, key, val):
    Q = dict(P); Q[key] = val
    if key == 'kappa':
        Q['T'] = Q['kappa']*Q['mu']
    if key in ('pc', 'egap', 'd', 'phi'):
        Q['K'] = round(Q['pc']*Q['egap']*Q['d']/1000 - Q['phi'], 3)
    return Q

# ------------------------------------------------------------- Monte Carlo
def mc_run(width, rho_val, seed=42, N=2000):
    """N uniform +/-width draws on alpha, beta, T, c_N, eta and the cost gap,
    common random numbers across designs; each structure at its own optimum."""
    rng = np.random.default_rng(seed)
    keys = ['alpha', 'beta', 'T', 'cN', 'eta']
    base = dict(alpha=P0['alpha'], beta=P0['beta'], T=P0['T'], cN=P0['cN'],
                eta=P0['eta'], gap=P0['cB'] - P0['cN'])
    U = rng.uniform(-1.0, 1.0, size=(N, 6))
    Wstars = np.empty(N); sBstars = np.empty(N); win = np.empty(N, dtype=int)
    for i in range(N):
        Q = dict(P0); Q['rho'] = rho_val
        for j, k in enumerate(keys):
            Q[k] = base[k]*(1.0 + width*U[i, j])
        Q['cB'] = Q['cN'] + base['gap']*(1.0 + width*U[i, 5])
        bm = best_menu_robust_fast(Q)
        if bm is None:
            bm = dict(W=-1e18, sB=np.nan)
        wN_ = poolingN(Q)['W']; wG_ = poolingG(Q)['W']
        ws = [bm['W'], wN_, wG_]
        w = int(np.argmax(ws)); win[i] = w
        Wstars[i] = ws[w]; sBstars[i] = bm['sB']
    return dict(share_menu=100*np.mean(win == 0), share_poolN=100*np.mean(win == 1),
                share_poolG=100*np.mean(win == 2), W=Wstars, sB=sBstars, win=win,
                cv=100*np.std(Wstars)/np.mean(Wstars))

MC_WIDTHS = (0.10, 0.20, 0.30)
MC_RHOS = (0.15, 0.30, 0.45)

def run_mc():
    mc = {}
    for w in MC_WIDTHS:
        for rv in MC_RHOS:
            print('  MC width=%.2f rho=%.2f ...' % (w, rv), flush=True)
            mc[(w, rv)] = mc_run(w, rv)
    np.savez(CACHE,
             **{'W_%g_%g' % k: v['W'] for k, v in mc.items()},
             **{'sB_%g_%g' % k: v['sB'] for k, v in mc.items()},
             **{'win_%g_%g' % k: v['win'] for k, v in mc.items()})
    return mc

def load_mc():
    if not os.path.exists(CACHE):
        return None
    z = np.load(CACHE)
    mc = {}
    for w in MC_WIDTHS:
        for rv in MC_RHOS:
            key = '%g_%g' % (w, rv)
            W = z['W_' + key]; sB = z['sB_' + key]; win = z['win_' + key]
            mc[(w, rv)] = dict(W=W, sB=sB, win=win,
                               share_menu=100*np.mean(win == 0),
                               share_poolN=100*np.mean(win == 1),
                               share_poolG=100*np.mean(win == 2),
                               cv=100*np.std(W)/np.mean(W))
    return mc

# ---------------------------------------------------------------- checking
def _check(name, got, want, tol=0.005):
    ok = abs(got - want) <= max(abs(want)*tol, tol)
    print('  %-46s %12.2f   reported %12.2f   %s' % (name, got, want,
                                                     'OK' if ok else '<-- CHECK'))
    return ok

def core_results():
    print('== Baseline calibration (Table: benchmark outcomes) ==')
    best = best_menu_robust_fast(P0)
    pN_ = poolingN(P0)
    pG_ = poolingG(P0)
    go = best_separating(P0)           # separating optimum (activation boundary)
    sBm = sBmin_locked(P0)
    opl = operator(P0, sBm, 0.0)
    W_lock = welfare(P0, sBm, 0.0, opl)
    # green-only peak on the s_N = 0 path
    grid = np.linspace(0, 400, 1601)
    Wg = np.array([welfare(P0, s, 0.0, operator(P0, s, 0.0))
                   if active(P0, s, 0.0) else -np.inf for s in grid])
    sG = grid[int(np.argmax(Wg))]
    res = minimize_scalar(lambda s: -welfare(P0, s, 0.0, operator(P0, s, 0.0)),
                          bounds=(sG - 1, sG + 1), method='bounded')
    sB_go = res.x
    op_go = operator(P0, sB_go, 0.0)
    W_go = welfare(P0, sB_go, 0.0, op_go)
    ok = True
    ok &= _check('menu s_B*', best['sB'], 65.57)
    ok &= _check('menu s_N*', best['sN'], 57.70)
    ok &= _check('menu p_B', best['op']['pB'], 431.38)
    ok &= _check('menu p_N', best['op']['pN'], 417.82)
    ok &= _check('menu price gap', best['op']['pB'] - best['op']['pN'], 13.57)
    ok &= _check('menu q_B', best['op']['qB'], 52.63)
    ok &= _check('menu q_N', best['op']['qN'], 121.29)
    ok &= _check('menu W', best['W'], 17998.70)
    ok &= _check('green-only s_B', sB_go, 184.63)
    ok &= _check('green-only price', op_go['pB'], 424.22)
    ok &= _check('green-only W', W_go, 17795.29)
    ok &= _check('locked s_B^min', sBm, 9.77)
    ok &= _check('locked p_N', opl['pN'], 446.67)
    ok &= _check('locked W', W_lock, 17234.29)
    ok &= _check('pooling (trad.) s_N', pN_['sN'], 57.89)
    ok &= _check('pooling (trad.) price', pN_['p'], 417.72)
    ok &= _check('pooling (trad.) q', pN_['q'], 173.42)
    ok &= _check('pooling (trad.) W', pN_['W'], 18044.18)
    ok &= _check('pooling (all-green) s_B', pG_['sB'], 61.61)
    ok &= _check('pooling (all-green) price', pG_['p'], 428.11)
    ok &= _check('pooling (all-green) q', pG_['q'], 164.58)
    ok &= _check('pooling (all-green) W  [corrected]', pG_['W'], 16102.77)
    # unsubsidized pooling
    pu0 = min((P0['alpha'] + P0['beta']*P0['cN'])/(2*P0['beta']), P0['pbar'])
    q0 = P0['alpha'] - P0['beta']*pu0
    W_pool0 = q0**2/(2*P0['beta']) + (pu0 - P0['cN'])*q0 + P0['K']*q0
    ok &= _check('pooling (trad.), unsubsidized W', W_pool0, 16913.13)
    ok &= _check('menu gain over unsubsidized pooling (%)',
                 100*(best['W']/W_pool0 - 1), 6.42, tol=0.05)

    print('\n== Welfare identity components at the separating optimum (Table: decomposition) ==')
    op = best['op']; b, r = P0['beta'], P0['rho']
    CSg = op['qB']**2/(2*r*b); CSt = op['qN']**2/(2*(1-r)*b)
    S = best['sB']*op['qB'] + best['sN']*op['qN']
    prof_incl = op['profit']
    ok &= _check('consumer surplus, green', CSg, 3077.37)
    ok &= _check('consumer surplus, traditional', CSt, 7005.53)
    ok &= _check('operator profit (incl. subsidy, net of F)', prof_incl, 20015.80)
    ok &= _check('K*(q_B+q_N)', P0['K']*(op['qB'] + op['qN']), 17.57)
    ok &= _check('eta*q_B', P0['eta']*op['qB'], 421.02)
    ok &= _check('-omega*S', -P0['omega']*S, -12538.59)
    ok &= _check('subsidy receipts S', S, 10448.82)
    ok &= _check('identity: sum = W',
                 CSg + CSt + prof_incl + P0['K']*(op['qB'] + op['qN'])
                 + P0['eta']*op['qB'] - P0['omega']*S, best['W'])

    print('\n== Critical values ==')
    ok &= _check('s_B^crit = c_B + A_B/beta - 2 pbar', scrit(P0), 28.33)
    ok &= _check('s_B^min', sBm, 9.77)

    print('\n== Arc elasticities at the separating optimum (+-1% perturbations) ==')
    elas = {}
    for key in ['alpha', 'omega', 'rho', 'kappa', 'eta', 'K', 'F']:
        v0 = P0[key]
        bp_, bm_ = with_param(P0, key, v0*1.01), with_param(P0, key, v0*0.99)
        Wp, Wm = best_separating(bp_)['W'], best_separating(bm_)['W']
        sp, sm = best_separating(bp_)['sB'], best_separating(bm_)['sB']
        eW = ((Wp - Wm)/((Wp + Wm)/2))/0.02
        es = ((sp - sm)/((sp + sm)/2))/0.02
        elas[key] = (eW, es)
        print('  %-6s eps_W = %+7.3f   eps_sB = %+7.3f' % (key, eW, es))
    ok &= _check('eps_W(alpha)', elas['alpha'][0], 6.178, tol=0.05)
    ok &= _check('eps_sB(alpha)', elas['alpha'][1], 2.617, tol=0.05)
    ok &= _check('eps_sB(omega)', elas['omega'][1], -5.880, tol=0.05)

    print('\n== Dominance thresholds (menu vs optimally subsidized pooling) ==')
    dW = lambda P: best_separating(P)['W'] - poolingN(P)['W']
    thr = lambda key, lo, hi: brentq(lambda v: dW(with_param(P0, key, v)), lo, hi,
                                     xtol=1e-8)
    t_eta = thr('eta', 8.0, 10.0); t_kap = thr('kappa', 0.15, 0.20)
    t_rho = thr('rho', 0.30, 0.55); t_om = thr('omega', 1.0, 1.19)
    ok &= _check('eta threshold', t_eta, 8.8635, tol=0.002)
    ok &= _check('kappa threshold', t_kap, 0.1572, tol=0.002)
    ok &= _check('rho threshold', t_rho, 0.4304, tol=0.002)
    ok &= _check('omega threshold', t_om, 1.1071, tol=0.002)
    f_al = lambda a: scrit(with_param(P0, 'alpha', a)) - \
        best_separating(with_param(P0, 'alpha', a))['sB']
    a_switch = brentq(f_al, 800, 900, xtol=1e-8)
    ok &= _check('alpha release->locked switch', a_switch, 829.81, tol=0.002)
    dWr = lambda a: best_menu_robust_fast(with_param(P0, 'alpha', a))['W'] - \
        poolingN(with_param(P0, 'alpha', a))['W']
    a_dom = brentq(dWr, 829.9, 850.0, xtol=1e-6)
    ok &= _check('alpha menu-dominance threshold', a_dom, 830.7, tol=0.002)

    print('\n== Tie-breaking robustness (Table: tiebreak) ==')
    W0g = None
    for eps in (0.0, 0.05, 0.10, 0.15):
        g = green_only_mixed(P0, eps)
        if W0g is None:
            W0g = g['W']
        print('  eps=%.2f  s_B*=%7.2f  p=%7.2f  W=%10.2f  Delta=%6.3f%%'
              % (eps, g['sB'], g['p'], g['W'], 100*(g['W']/W0g - 1)))
    g = green_only_mixed(P0, 0.15)
    ok &= _check('tie: s_B at eps=0.15', g['sB'], 141.61)
    ok &= _check('tie: welfare loss at eps=0.15 (%)',
                 100*(1 - g['W']/W0g), 1.221, tol=0.05)

    return dict(best=best, pN_=pN_, pG_=pG_, W_lock=W_lock, sBm=sBm, opl=opl,
                sB_go=sB_go, op_go=op_go, W_go=W_go, W_pool0=W_pool0, elas=elas,
                thr=dict(eta=t_eta, kappa=t_kap, rho=t_rho, omega=t_om,
                         alpha_switch=a_switch, alpha_dom=a_dom), ok=ok)

def green_path(P, n=1201):
    sB_path = np.linspace(0, 300, n)
    W_path = np.full(n, np.nan)
    PB = np.full(n, np.nan); PN = np.full(n, np.nan)
    QB = np.full(n, np.nan); QN = np.full(n, np.nan)
    for i, s in enumerate(sB_path):
        o = operator(P, s, 0.0)
        if o['profit'] >= fallback_profit(P, 0.0) - 1e-9:
            W_path[i] = welfare(P, s, 0.0, o)
            PB[i], PN[i], QB[i], QN[i] = o['pB'], o['pN'], o['qB'], o['qN']
    return sB_path, W_path, PB, PN, QB, QN

def mc_check(mc):
    print('\n== Monte Carlo (seed 42, N=2000, common random numbers) ==')
    exp = {(0.10, 0.15): 26.0, (0.10, 0.30): 51.3, (0.10, 0.45): 66.4,
           (0.20, 0.15): 28.3, (0.20, 0.30): 46.8, (0.20, 0.45): 55.7,
           (0.30, 0.15): 28.3, (0.30, 0.30): 39.9, (0.30, 0.45): 45.9}
    ok = True
    for k, v in sorted(exp.items()):
        got = mc[k]['share_menu']
        flag = 'OK' if abs(got - v) < 0.15 else '<-- CHECK'
        print('  width=%.2f rho=%.2f  separating share %5.1f%% (reported %4.1f%%)  %s'
              % (k[0], k[1], got, v, flag))
        ok &= flag == 'OK'
    for w in MC_WIDTHS:
        cv = mc[(w, 0.30)]['cv']
        print('  width=%.2f  CV of W* = %.1f%%' % (w, cv))
    r = mc[(0.10, 0.30)]
    q1, q2, q3 = np.percentile(r['sB'], [25, 50, 75])
    print('  s_B* distribution (rho=0.30, +-10%%): IQR [%.1f, %.1f], median %.1f'
          % (q1, q3, q2))
    return ok

# ------------------------------------------------------------------ figures
C = dict(blue='#2166ac', red='#b2182b', green='#1b9e77', orange='#e08214',
         gray='#666666', lblue='#92c5de', lred='#f4a582', purple='#762a83')

def _mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
                         'figure.dpi': 200, 'savefig.dpi': 200})
    return plt

def fig1(res):
    plt = _mpl()
    best, pN_, pG_ = res['best'], res['pN_'], res['pG_']
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4),
                             gridspec_kw={'width_ratios': [1.0, 1.35]})
    # (a) horizontal dot plot with the separating menu as reference
    ax = axes[0]
    names = ['Pooling\n(all-green)', 'Pooling\n(trad.)', 'Separating,\nlocked',
             'Separating,\ngreen-only', 'Separating\n(release)']
    vals = [pG_['W'], pN_['W'], res['W_lock'], res['W_go'], best['W']]
    cols = [C['lred'], C['orange'], C['lblue'], C['lblue'], C['blue']]
    ypos = np.arange(len(vals))
    x0 = 15800.0
    ax.axvline(best['W'], color=C['gray'], lw=0.9, ls='--', zorder=1)
    for y, v, c in zip(ypos, vals, cols):
        ax.plot([x0, v], [y, y], color=c, lw=1.2, zorder=2)
        ax.plot(v, y, 'o', ms=8, color=c, mec='black', mew=0.6, zorder=3)
        ax.text(v + 55, y, '{:,.2f}'.format(v), va='center', fontsize=7.5)
    ax.text(best['W'] - 40, 4.42, 'separating menu\n(reference)', fontsize=7,
            color=C['gray'], ha='right', va='top')
    ax.set_yticks(ypos); ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlim(x0, 18500); ax.set_ylim(-0.6, 4.7)
    ax.set_xlabel('maximized welfare ($10^4$ CNY per year)')
    ax.set_title('(a) Welfare by market structure', loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    # (b) horizontal bridge of the welfare identity
    ax = axes[1]
    op = best['op']; b, r = P0['beta'], P0['rho']
    comp = [('CS$_G$', op['qB']**2/(2*r*b)),
            ('CS$_P$', op['qN']**2/(2*(1-r)*b)),
            ('Profit$^{\\dagger}$', op['profit']),
            ('$K(q_B+q_N)$', P0['K']*(op['qB'] + op['qN'])),
            ('$\\eta q_B$', P0['eta']*op['qB']),
            ('$-\\omega S$', -P0['omega']*(best['sB']*op['qB'] + best['sN']*op['qN']))]
    cum = 0.0
    for i, (lab, v) in enumerate(comp):
        y = len(comp) - i
        left = cum if v >= 0 else cum + v
        ax.barh(y, abs(v), left=left, height=0.62,
                color=C['blue'] if v >= 0 else C['red'], edgecolor='black', lw=0.4)
        ax.text(left + abs(v) + 300, y, '{:+,.0f}'.format(v), va='center', fontsize=7)
        if i < len(comp) - 1:
            ax.plot([cum + v, cum + v], [y - 0.31, y - 1 + 0.31], color='gray', lw=0.5, ls=':')
        cum += v
    ax.plot([cum, cum], [0.69, 1 - 0.31], color='gray', lw=0.5, ls=':')
    ax.barh(0, cum, height=0.62, color=C['green'], edgecolor='black', lw=0.4)
    ax.text(cum + 300, 0, '{:,.0f}'.format(cum), va='center', fontsize=7.5)
    ax.set_yticks(list(range(len(comp) + 1)))
    ax.set_yticklabels(['$W$'] + [c[0] for c in comp[::-1]], fontsize=7.5)
    ax.set_xlim(-1000, 34500); ax.set_ylim(-0.65, len(comp) + 0.65)
    ax.set_xlabel('$10^4$ CNY per year')
    ax.set_title('(b) Welfare identity at the separating optimum', loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'Fig1.png'))
    plt.close(fig)

def operator_grid_regime(P, sBm, sNm):
    """Vectorized best response returning regime id (0=I,1=IIa,2=IIb,3=III,4=IV)."""
    AB, AN, MB, MN = margins(P)
    b, pb, T, r, F = P['beta'], P['pbar'], P['T'], P['rho'], P['F']
    cB, cN = P['cB'], P['cN']
    pBu = (AB + b*(cB - sBm))/(2*b); pNu = (AN + b*(cN - sNm))/(2*b)
    def prof(pB, pN):
        return ((pB + sBm - cB)*r*np.maximum(AB - b*pB, 0)
                + (pN + sNm - cN)*(1-r)*np.maximum(AN - b*pN, 0) - F)
    best = np.full(sBm.shape, -1e18); reg = np.full(sBm.shape, -1)
    bpB = np.zeros_like(best); bpN = np.zeros_like(best)
    def upd(pB, pN, rid, mask):
        nonlocal best, reg, bpB, bpN
        ok = (mask & (pB >= -1e-9) & (pN >= -1e-9) & (pB <= pb + 1e-9) & (pN <= pb + 1e-9)
              & (pB - pN >= -1e-9) & (pB - pN <= T + 1e-9))
        v = np.where(ok, prof(np.clip(pB, 0, pb), np.clip(pN, 0, pb)), -1e18)
        imp = v > best + 1e-12
        best = np.where(imp, v, best); reg = np.where(imp, rid, reg)
        bpB = np.where(imp, pB, bpB); bpN = np.where(imp, pN, bpN)
    upd(pBu, pNu, 0, np.ones_like(best, bool))
    pN2 = (r*AB + (1-r)*AN - r*b*(sBm - cB) - (1-r)*b*(sNm - cN) - 2*r*b*T)/(2*b)
    upd(pN2 + T, pN2, 1, np.ones_like(best, bool))
    pU = (r*AB + (1-r)*AN + r*b*(cB - sBm) + (1-r)*b*(cN - sNm))/(2*b)
    upd(pU, pU, 2, np.ones_like(best, bool))
    pN3 = np.clip(pNu, pb - T, pb)
    upd(np.full_like(best, pb), pN3, np.where(np.abs(pN3 - (pb - T)) < 1e-9, 4, 3),
        pBu > pb)
    upd(np.full_like(best, pb), np.full_like(best, pb), 3, (pBu > pb) | (pNu > pb))
    return bpB, bpN, best, reg

def fig2(res):
    plt = _mpl()
    sBs = np.linspace(0, 200, 241); sNs = np.linspace(0, 100, 161)
    SBm, SNm = np.meshgrid(sBs, sNs)
    pB, pN, pi, reg = operator_grid_regime(P0, SBm, SNm)
    pfb = np.minimum((P0['alpha'] + P0['beta']*(P0['cN'] - SNm))/(2*P0['beta']), P0['pbar'])
    pi0 = (pfb + SNm - P0['cN'])*np.maximum(P0['alpha'] - P0['beta']*pfb, 0)
    actv = pi >= pi0 - 1e-9
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    # (a) categorical regime map with in-region labels and hatched inactive side
    ax = axes[0]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#2166ac', '#92c5de', '#1b9e77', '#e08214', '#b2182b'])
    ax.pcolormesh(sBs, sNs, reg, cmap=cmap, shading='auto', vmin=-0.5, vmax=4.5, alpha=0.85)
    with plt.rc_context({'hatch.color': 'white', 'hatch.linewidth': 0.55}):
        ax.contourf(sBs, sNs, (~actv).astype(float), levels=[0.5, 1.5], colors='none',
                    hatches=['//////'])
    ax.contour(sBs, sNs, pi - pi0, levels=[0.0], colors='black', linewidths=1.4)
    jtop = int(np.argmin(np.abs((pi - pi0)[-1, :])))
    ax.annotate('activation boundary', (sBs[jtop], 99), xytext=(min(sBs[jtop] + 26, 150), 91),
                fontsize=7.5, arrowprops=dict(arrowstyle='-', lw=0.6))
    for rid, lab in {0: 'I', 1: 'IIa', 2: 'IIb', 3: 'III', 4: 'IV'}.items():
        m = reg == rid
        if m.sum() == 0:
            continue
        ys, xs = np.where(m)
        ax.text(sBs[xs].mean(), sNs[ys].mean(), lab, fontsize=11, fontweight='bold',
                color='black', ha='center', va='center')
    ax.plot(65.57, 57.70, marker='*', ms=13, color='gold', mec='black', mew=0.6, zorder=5)
    ax.plot(9.77, 0.0, marker='s', ms=7, color='white', mec='black', mew=0.7, zorder=5)
    ax.plot(184.63, 0.0, marker='^', ms=8, color='white', mec='black', mew=0.7, zorder=5)
    ax.set_xlabel('$s_B$ (CNY per ton)'); ax.set_ylabel('$s_N$ (CNY per ton)')
    ax.set_title('(a) KKT regimes and the activation boundary', loc='left')
    # (b) labeled contour map of the price gap
    ax = axes[1]
    gap = np.clip(pB - pN, 0, P0['T'])
    cf = ax.contourf(sBs, sNs, gap, levels=np.linspace(0, P0['T'], 16), cmap='cividis')
    cl = ax.contour(sBs, sNs, gap, levels=[2, 4, 6, 8, 10, 12, 14], colors='white',
                    linewidths=0.5)
    ax.clabel(cl, fmt='%d', fontsize=6.5)
    ax.contour(sBs, sNs, pi - pi0, levels=[0.0], colors='red', linewidths=1.4,
               linestyles='--')
    ax.annotate('activation boundary', (sBs[jtop], 99), xytext=(min(sBs[jtop] + 26, 150), 91),
                fontsize=7.5, color='red', arrowprops=dict(arrowstyle='-', lw=0.6, color='red'))
    fig.colorbar(cf, ax=ax, label='$p_B - p_N$ (CNY per ton)')
    ax.plot(65.57, 57.70, marker='*', ms=13, color='red', mec='black', mew=0.6, zorder=5)
    ax.set_xlabel('$s_B$ (CNY per ton)'); ax.set_ylabel('$s_N$ (CNY per ton)')
    ax.set_title('(b) Equilibrium price gap (contours)', loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'Fig2.png'))
    plt.close(fig)

def fig3(res):
    plt = _mpl()
    panels = [('alpha', np.linspace(600, 1000, 161), r'$\alpha$', 800.0, res['thr']['alpha_dom']),
              ('omega', np.linspace(0.90, 1.50, 161), r'$\omega$', 1.2, res['thr']['omega']),
              ('rho', np.linspace(0.20, 0.50, 161), r'$\rho$', 0.30, res['thr']['rho']),
              ('kappa', np.linspace(0.11, 0.19, 161), r'$\kappa$', 0.15, res['thr']['kappa']),
              ('eta', np.linspace(6.0, 10.0, 161), r'$\eta$', 8.0, res['thr']['eta'])]
    fig, axes = plt.subplots(1, 5, figsize=(10.4, 2.6))
    for ax, (key, xs, lab, base, thr) in zip(axes, panels):
        sBs_ = np.full_like(xs, np.nan); dW_ = np.full_like(xs, np.nan)
        for i, v in enumerate(xs):
            Q = with_param(P0, key, v)
            bs = best_separating(Q)
            sBs_[i] = bs['sB']
            dW_[i] = bs['W'] - poolingN(Q)['W']
        ax.plot(xs, sBs_, color=C['blue'], lw=1.3)
        ax.axvline(base, color='gray', lw=0.7, ls=':')
        ax.set_xlabel(lab)
        ax2 = ax.twinx()
        ax2.plot(xs, dW_, color=C['red'], lw=1.1, ls='--')
        ax2.axhline(0, color=C['red'], lw=0.5, ls=':')
        ax2.plot([thr], [0.0], 'o', ms=4.5, color=C['red'])
        if key == 'alpha':
            ax.set_ylabel('$s_B^*$ (CNY per ton)', color=C['blue'])
        if key == 'eta':
            ax2.set_ylabel('$W^{sep}-W^{pool}$', color=C['red'])
        ax.tick_params(axis='y', labelcolor=C['blue'])
        ax2.tick_params(axis='y', labelcolor=C['red'])
    fig.suptitle('One-factor sensitivity: $s_B^*$ (blue, left) and menu advantage (red dashed, right)',
                 y=1.02, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'Fig3.png'), bbox_inches='tight')
    plt.close(fig)

def fig5(res):
    plt = _mpl()
    sB_path, W_path, PB, PN, QB, QN = green_path(P0)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.2))
    ax = axes[0]
    ax.axvspan(0, res['sBm'], color='0.92', zorder=0)
    ax.plot(sB_path, PB, color=C['blue'], lw=1.4, label='$p_B$')
    ax.plot(sB_path, PN, color=C['lblue'], lw=1.4, ls='--', label='$p_N$')
    ax2 = ax.twinx()
    ax2.plot(sB_path, PB - PN, color=C['green'], lw=1.2, label='$p_B-p_N$')
    ax2.axhline(P0['T'], color=C['green'], lw=0.8, ls=':')
    ax2.set_ylabel('price gap (CNY per ton)', color=C['green'])
    ax2.tick_params(axis='y', labelcolor=C['green'])
    for x, lab, yt in [(res['sBm'], '$s_B^{\\min}$', 448.5),
                       (scrit(P0), '$s_B^{\\mathrm{crit}}$', 443.5),
                       (P0['cB'] - P0['cN'] + P0['T'], '$c_B-c_N+T$', 438.5)]:
        ax.axvline(x, color='gray', lw=0.8, ls=':')
        ax.text(x + 1.5, yt, lab, fontsize=7.5, rotation=90, va='top')
    ax.set_xlabel('$s_B$ (CNY per ton, with $s_N=0$)'); ax.set_ylabel('price (CNY per ton)')
    ax.legend(loc='center right', fontsize=8, frameon=False)
    ax.set_title('(a) Menu prices and the price gap', loc='left')
    ax = axes[1]
    ax.axvspan(0, res['sBm'], color='0.92', zorder=0)
    ax.plot(sB_path, QB, color=C['blue'], lw=1.4, label='$q_B$')
    ax.plot(sB_path, QN, color=C['orange'], lw=1.4, label='$q_N$')
    ax.plot(sB_path, QB + QN, color='black', lw=1.2, ls='--', label='$q_B+q_N$')
    for x in (res['sBm'], scrit(P0), P0['cB'] - P0['cN'] + P0['T']):
        ax.axvline(x, color='gray', lw=0.8, ls=':')
    ax.text(4, 165, 'green service\nwithdrawn', fontsize=7, ha='center')
    ax.set_xlabel('$s_B$ (CNY per ton, with $s_N=0$)'); ax.set_ylabel('quantity ($10^4$ tons per year)')
    ax.legend(loc='center right', fontsize=8, frameon=False)
    ax.set_title('(b) Quantities', loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'Fig5.png'))
    plt.close(fig)

def policy_map():
    rho_g = np.linspace(0.05, 0.60, 24)
    kap_g = np.linspace(0.02, 0.30, 24)
    POL = np.zeros((len(kap_g), len(rho_g)), dtype=int)   # 0 menu, 1 poolN, 2 poolG
    for a_, rv in enumerate(rho_g):
        for b_, kv in enumerate(kap_g):
            Q = with_param(with_param(P0, 'rho', rv), 'kappa', kv)
            bm = best_menu_robust_fast(Q)
            bmW = bm['W'] if bm is not None else -1e18
            POL[b_, a_] = int(np.argmax([bmW, poolingN(Q)['W'], poolingG(Q)['W']]))
    return rho_g, kap_g, POL

def fig4(res, mc):
    plt = _mpl()
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    r1030 = mc[(0.10, 0.30)]
    sB_path, W_path, PB, PN, QB, QN = green_path(P0)
    rho_g, kap_g, POL = policy_map()
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    ax = axes[0, 0]
    params_ = ['alpha', 'omega', 'rho', 'kappa', 'eta', 'K', 'F']
    labels_ = [r'$\alpha$', r'$\omega$', r'$\rho$', r'$\kappa$', r'$\eta$', '$K$', '$F$']
    eW = [res['elas'][k][0] for k in params_]; es = [res['elas'][k][1] for k in params_]
    x = np.arange(len(params_))
    ax.bar(x - 0.19, eW, width=0.38, color=C['blue'], label=r'$\varepsilon_W$')
    ax.bar(x + 0.19, es, width=0.38, color=C['orange'], label=r'$\varepsilon_{s_B}$')
    ax.set_xticks(x); ax.set_xticklabels(labels_)
    ax.axhline(0, color='black', lw=0.6)
    ax.set_ylabel('arc elasticity'); ax.legend(frameon=False, loc='lower left')
    ax.set_title('(a) Standardized elasticities at the separating optimum', loc='left')
    ax.set_ylim(-7.5, 7.5)
    ax = axes[0, 1]
    cmap2 = ListedColormap([C['blue'], '#d9d9d9', C['orange']])
    ax.pcolormesh(rho_g, kap_g, POL, cmap=cmap2, shading='auto', vmin=-0.5, vmax=2.5)
    ax.plot(0.30, 0.15, marker='*', ms=14, color=C['red'], mec='black', mew=0.5, zorder=5)
    ax.annotate('benchmark (0.30, 0.15)', (0.30, 0.15), xytext=(0.33, 0.09), fontsize=8,
                arrowprops=dict(arrowstyle='-', lw=0.6))
    ax.set_xlabel(r'$\rho$ (green share)'); ax.set_ylabel(r'$\kappa$ (premium coefficient)')
    ax.set_title('(b) Optimal market structure', loc='left')
    hs = [Patch(fc=C['blue'], label='separating menu'),
          Patch(fc='#d9d9d9', label='pooling (traditional)'),
          Patch(fc=C['orange'], label='pooling (all-green)')]
    ax.legend(handles=hs, loc='lower right', fontsize=7.5, framealpha=0.95)
    ax = axes[1, 0]
    ax.hist(r1030['W'], bins=60, color=C['lblue'], edgecolor='white', lw=0.3)
    ax.axvline(res['best']['W'], color=C['red'], lw=1.2, ls='--')
    ax.text(res['best']['W'] + 800, ax.get_ylim()[1]*0.9, 'benchmark', color=C['red'],
            fontsize=7.5)
    ax.set_xlabel('maximized welfare $W^*$ ($10^4$ CNY per year)'); ax.set_ylabel('frequency')
    ax.set_title('(c) Monte Carlo distribution of $W^*$ ($\\pm$10%, $\\rho$=0.30)',
                 loc='left')
    ax = axes[1, 1]
    ax.set_xlim(0, 300); ax.set_ylim(16600, 18350)
    ax.axvspan(0, res['sBm'], color='0.92', zorder=0)
    ax.plot(sB_path, W_path, color=C['blue'], lw=1.4, zorder=3)
    ax.plot([0, res['sBm']], [res['W_pool0'], res['W_pool0']], color=C['blue'], lw=1.0,
            ls=':', zorder=3)
    ax.axhline(res['pN_']['W'], color=C['red'], lw=1.0, ls='--', zorder=2)
    ax.axvline(res['sBm'], color='gray', lw=0.8, ls=':')
    ax.axvline(scrit(P0), color='gray', lw=0.8, ls=':')
    ax.plot(res['sB_go'], res['W_go'], marker='^', ms=8, color=C['green'], mec='black',
            mew=0.5, zorder=5)
    ax.text(11, 16720, '$s_B^{\\min}$', fontsize=7.5)
    ax.text(29.5, 16720, '$s_B^{\\mathrm{crit}}$', fontsize=7.5)
    ax.text(240, 18100, 'optimally subsidized pooling (18,044)', fontsize=7.5,
            color=C['red'], ha='center')
    ax.text(4.9, 17000, 'green service\nwithdrawn', fontsize=7, ha='center')
    ax.annotate('green-only peak (%.2f)' % res['sB_go'], (res['sB_go'], res['W_go']),
                xytext=(150, 17050), fontsize=7.5,
                arrowprops=dict(arrowstyle='-', lw=0.6))
    ax.set_xlabel('$s_B$ (CNY per ton, with $s_N=0$)'); ax.set_ylabel('welfare ($10^4$ CNY per year)')
    ax.set_title('(d) Welfare along the green-only subsidy path', loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'Fig4.png'))
    plt.close(fig)

def fig6(res, mc):
    plt = _mpl()
    r1030 = mc[(0.10, 0.30)]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.2))
    ax = axes[0]
    sB_menu = r1030['sB'][r1030['win'] == 0]
    sB_pool = r1030['sB'][r1030['win'] != 0]
    bins = np.linspace(0, 300, 61)
    ax.hist([sB_menu, sB_pool], bins=bins, stacked=True,
            color=[C['blue'], '#d9d9d9'], edgecolor='white', lw=0.3,
            label=['menu optimal (%.1f%%)' % r1030['share_menu'],
                   'pooling optimal (%.1f%%)' % (100 - r1030['share_menu'])])
    ax.axvline(res['best']['sB'], color=C['red'], lw=1.2, ls='--')
    ax.text(res['best']['sB'] + 4, ax.get_ylim()[1]*0.9, 'benchmark $s_B^*$',
            color=C['red'], fontsize=7.5)
    ax.set_xlabel('$s_B^*$ (CNY per ton)'); ax.set_ylabel('frequency')
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title('(a) Optimal green subsidy across MC draws ($\\pm$10%, $\\rho$=0.30)',
                 loc='left')
    ax = axes[1]
    shares = [mc[(0.10, rv)]['share_menu'] for rv in MC_RHOS]
    N = 2000
    ci = [1.96*np.sqrt(s/100*(1 - s/100)/N)*100 for s in shares]
    ax.bar(range(3), shares, yerr=ci, capsize=4, color=C['lblue'], edgecolor='black',
           lw=0.4)
    for i, s in enumerate(shares):
        ax.text(i, s + 2.5, '%.1f%%' % s, ha='center', fontsize=8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['$\\rho$=0.15', '$\\rho$=0.30', '$\\rho$=0.45'])
    ax.set_ylabel('separating-menu share (%)'); ax.set_ylim(0, 80)
    ax.set_title('(b) Separating-menu share by green share (95% CI)', loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'Fig6.png'))
    plt.close(fig)

# --------------------------------------------------------------------- main
def main():
    res = core_results()
    fig1(res); print('\nFig1 written.')
    fig2(res); print('Fig2 written.')
    fig3(res); print('Fig3 written.')
    fig5(res); print('Fig5 written.')
    mc = None
    if '--mc' in sys.argv:
        print('\nRunning Monte Carlo study (9 designs x 2000 draws) ...')
        mc = run_mc()
    else:
        mc = load_mc()
    if mc is None:
        print('\nFigures 4 and 6 need Monte Carlo results: run once with --mc,'
              ' or provide figs/mc_cache.npz.')
    else:
        mc_check(mc)
        fig4(res, mc); print('Fig4 written.')
        fig6(res, mc); print('Fig6 written.')
    print('\n%s' % ('ALL CHECKS PASSED.' if res['ok'] else 'SOME CHECKS NEED REVIEW.'))

if __name__ == '__main__':
    main()
