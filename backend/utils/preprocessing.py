import numpy as np


def filter_low_variance_features(X: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    variances = np.var(X, axis=0)
    return np.where(variances >= threshold)[0]


def filter_high_correlation_features(X: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    corr = np.corrcoef(X.T)
    keep = np.ones(X.shape[1], dtype=bool)
    for i in range(X.shape[1]):
        if keep[i]:
            for j in range(i + 1, X.shape[1]):
                if abs(corr[i, j]) > threshold:
                    keep[j] = False
    return np.where(keep)[0]
