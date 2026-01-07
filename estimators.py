import numpy as np
from Environment import MCEnv
from typing import Tuple



def solve_ground_truth(P: np.ndarray, R: np.ndarray, gamma: float) -> np.ndarray:
    
    n = P.shape[0]
    A = np.eye(n) - gamma * P
    return np.linalg.solve(A, R)

def uvn_estimate(env: MCEnv, i: int, M: int, gamma: float) -> Tuple[float, int]:
    """
    Ulam/von Neumann estimator (terminal reward scaled by 1/(1-gamma)):
      - sample stopping time tau ~ Geom(1-gamma): P(tau=k) = (1-gamma)*gamma^k
      - simulate chain up to tau
      - output Z = R(X_tau)/(1-gamma) 
    """
    assert 0 < gamma < 1
    rng = env.rng
    total = 0.0
    steps = 0
    for _ in range(M):
        x = i
        r_x_tau = env.R[x]
        k = 0
        # sampling tau using the first success method
        while True:
            # stop with prob 1-gamma
            if rng.random() < (1.0 - gamma):
                break
            x, r_x_tau = env.step(x)
            steps += 1
            k += 1
        # terminal reward at X_tau
        total += r_x_tau / (1.0 - gamma)
    return total / M, steps


def wasow_independent(env: MCEnv, i: int, M: int, gamma: float, N: int) -> Tuple[float, int]:
    """
    Wasow estimator with fixed horizon N and independent trajectories.
    """
    total = 0.0
    steps = 0
    for _ in range(M):
        x = i
        r_x = env.R[x]
        total += r_x  # k=0
        g = gamma
        for _k in range(1, N + 1):
            x, r_x = env.step(x)
            steps += 1
            total += g * r_x
            g *= gamma
    return total / M, steps


def reconciled_reuse_one_long_path(env: MCEnv, i: int, M: int, gamma: float, L: int) -> Tuple[float, int]:
    """
    Reconciled estimator implemented as trajectory reuse:
    For each sample m, generate one long trajectory of length L and reuse all prefixes.
    The raw formula matches Wasow:
      sum_{k=0}^L gamma^k R(X_k)
    """
    total = 0.0
    steps = 0
    for _ in range(M):
        x = i
        r_x = env.R[x]
        total += r_x 
        g = gamma
        for _k in range(1, L + 1):
            x, r_x = env.step(x)
            steps += 1
            total += g * r_x
            g *= gamma
    # same numerical form as Wasow with N=L, but we keep it separate for experiment semantics.
    return total / M, steps


def jacobi_evaluation(P: np.ndarray, R: np.ndarray, gamma: float, tol: float = 1e-8, max_iter: int = 20000) -> Tuple[np.ndarray, int]:
    """
    V^{k+1} = R + gamma P V^k until ||V^{k+1}-V^k||_inf < tol
    Returns V, number of iterations.
    """
    n = P.shape[0]
    V = np.zeros(n, dtype=float)
    it = 0
    while it < max_iter:
        V_new = R + gamma * (P @ V)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, it + 1
        V = V_new
        it += 1
    return V, it
