from __future__ import annotations

import numpy as np


def augment_with_agn(
    X,
    y,
    sigma: float = 0.01,
    n_aug: int = 9,
    mu: float = 0.0,
    random_state: int = 42,
    include_original: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if n_aug < 0:
        raise ValueError("n_aug must be non-negative")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)

    features = [X] if include_original else []
    labels = [y] if include_original else []
    for _ in range(n_aug):
        noise = rng.normal(mu, sigma, size=X.shape)
        features.append(X + noise)
        labels.append(y.copy())

    return np.vstack(features), np.concatenate(labels)

