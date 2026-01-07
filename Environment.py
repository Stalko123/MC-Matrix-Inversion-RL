import numpy as np 

from typing import Tuple

class MCEnv:
    """
    Markov chain environment with (optional) stochastic rewards.

    - Transition: X_{t+1} ~ P[X_t, :]
    - Reward emitted at time t is r_{t+1} with E[r_{t+1} | X_t=i] = R[i]

    We implement reward as r = R[x] + noise, i.e. reward depends on current state.
    """
    def __init__(self, P: np.ndarray, R: np.ndarray, reward_noise_std: float = 0.0, seed: int = 0):
        P = np.asarray(P, dtype=float)
        R = np.asarray(R, dtype=float)
        assert P.ndim == 2 and P.shape[0] == P.shape[1]
        assert R.ndim == 1 and R.shape[0] == P.shape[0]
        assert np.all(P >= 0), "P must be nonnegative"
        row_sums = P.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-10), "P must be stochastic"

        self.P = P
        self.R = R
        self.n = P.shape[0]
        self.noise = reward_noise_std
        self.rng = np.random.default_rng(seed)

    def step(self, x: int) -> Tuple[int, float]:
        x_next = self.rng.choice(self.n, p=self.P[x])
        r = self.R[x] + float(self.rng.normal(0.0, self.noise))
        return x_next, r
