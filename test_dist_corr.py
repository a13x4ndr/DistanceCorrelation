import numpy as np
from dist_corr import distance_correlation, distance_covariance

def test_perfect_correlation():
    X = np.random.rand(10, 50)
    Y = X.copy()
    R = distance_correlation(X, Y)
    assert np.isclose(R, 1.0, atol=1e-10)

def test_independent_fields():
    X = np.random.randn(10, 50)
    Y = np.random.randn(10, 50)
    R = distance_correlation(X, Y)
    assert 0.0 <= R <= 0.5  # Not perfectly correlated

def test_invalid_shapes():
    X = np.random.rand(10, 50)
    Y = np.random.rand(8, 50)  # mismatched samples
    with pytest.raises(ValueError):
        distance_correlation(X, Y)

def test_invalid_beta():
    X = np.random.rand(10, 50)
    Y = X.copy()
    with pytest.raises(ValueError):
        distance_correlation(X, Y, beta=2.5)

def test_zero_variance():
    X = np.ones((10, 50))
    Y = np.random.rand(10, 50)
    R = distance_correlation(X, Y)
    assert R == 0.0
