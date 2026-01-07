import numpy as np


def _normalize_rows(A: np.ndarray) -> np.ndarray:
    A = np.maximum(A, 0.0)
    s = A.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return A / s


def gen_dense_dirichlet(n: int, alpha: float = 1.0, seed: int = 0) -> np.ndarray:
    """
    Dense Markov chain: each row sampled i.i.d. from Dirichlet(alpha,...,alpha).
    """
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(alpha=np.full(n, alpha), size=n)
    return P


def gen_sparse_neighbor_chain(n: int, degree: int = 5, seed: int = 0) -> np.ndarray:
    """
    Sparse chain: each state transitions to 'degree' random neighbors (incl. itself),
    with random weights, then row-normalized.
    """
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        neighbors = rng.choice(n, size=min(degree, n), replace=False)
        if i not in neighbors:
            neighbors[0] = i
        weights = rng.random(len(neighbors))
        A[i, neighbors] = weights
    return _normalize_rows(A)


def gen_nearly_deterministic(n: int, p_main: float = 0.98, seed: int = 0) -> np.ndarray:
    """
    Nearly deterministic: each row has one dominant next state with prob p_main,
    remaining mass spread uniformly across others.
    """
    rng = np.random.default_rng(seed)
    P = np.zeros((n, n), dtype=float)
    for i in range(n):
        j_star = int(rng.integers(0, n))
        P[i, :] = (1.0 - p_main) / (n - 1) if n > 1 else 1.0
        P[i, j_star] = p_main
    return P


def gen_reducible_with_absorbing(n: int, absorb_frac: float = 0.1, seed: int = 0) -> np.ndarray:
    """
    Non-ergodic example: some absorbing states.
    """
    rng = np.random.default_rng(seed)
    P = gen_sparse_neighbor_chain(n, degree=5, seed=seed + 123)
    m_abs = max(1, int(round(absorb_frac * n)))
    absorbing_states = rng.choice(n, size=m_abs, replace=False)
    for s in absorbing_states:
        P[s, :] = 0.0
        P[s, s] = 1.0
    return P