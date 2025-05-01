import numpy as np


def l2_matrix(Z, beta=1.0):
    """
    Pairwise L2 distance matrix raised to power beta, normalized over spatial points.

    Parameters
    ----------
    Z : ndarray of shape (n_samples, n_points)
    beta : float in (0, 2)

    Returns
    -------
    D : ndarray of shape (n_samples, n_samples)
    """
    Z = np.asarray(Z)
    if Z.ndim != 2:
        raise ValueError("Input Z must be a 2D array (n_samples, n_points).")
    if not (0 < beta < 2):
        raise ValueError("beta must be in the open interval (0, 2).")

    n = Z.shape[0]
    D = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            diff = Z[i] - Z[j]
            D[i, j] = np.mean(diff ** 2) ** 0.5
    return D ** beta


def distance_covariance(X, Y, beta=1.0):
    """
    Compute the sample distance covariance T_{n,β}(X, Y) for random fields.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_points)
    Y : ndarray of shape (n_samples, n_points)
    beta : float in (0, 2)

    Returns
    -------
    T : float
        Sample distance covariance
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape.")
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays (n_samples, n_points).")
    if X.shape[0] < 2:
        raise ValueError("At least two samples are required.")
    if not (0 < beta < 2):
        raise ValueError("beta must be in the open interval (0, 2).")

    A = l2_matrix(X, beta)
    B = l2_matrix(Y, beta)

    n = X.shape[0]
    term1 = np.mean(A * B)
    term2 = np.mean(A) * np.mean(B)
    term3 = 2 * np.mean([np.mean(A[i]) * np.mean(B[j]) for i in range(n) for j in range(n)])

    return term1 + term2 - term3 / n


def distance_correlation(X, Y, beta=1.0):
    """
    Compute the sample distance correlation R_{n,β}(X, Y) based on equation (1.10).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_points)
    Y : ndarray of shape (n_samples, n_points)
    beta : float in (0, 2)

    Returns
    -------
    R : float
        Sample distance correlation
    """
    T_xy = distance_covariance(X, Y, beta)
    T_xx = distance_covariance(X, X, beta)
    T_yy = distance_covariance(Y, Y, beta)

    if T_xx > 0 and T_yy > 0:
        return T_xy / np.sqrt(T_xx * T_yy)
    else:
        return 0.0
