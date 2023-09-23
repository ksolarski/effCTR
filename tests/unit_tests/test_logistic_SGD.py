"""Test logistic SGD."""
import pytest
import scipy.sparse
import numpy as np
from effCTR.models.Logistic_SGD import Logistic_SGD


@pytest.fixture(scope="function")
def y_sparse():
    """Create a sample dataset."""
    y = np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0]).reshape(-1, 1)
    y = scipy.sparse.csr_matrix(y)
    return y


@pytest.fixture(scope="function")
def X_sparse():
    """Create a sample dataset."""
    X = np.array([0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1]).reshape(
        10, 2
    )
    X = scipy.sparse.csr_matrix(X)
    return X


def test_simple_logistic_SGD(X_sparse, y_sparse):
    """Test logistic SGD on a very simple case."""
    logistic_SGD = Logistic_SGD(chunksize=2)
    mod = logistic_SGD.fit(X_sparse, y_sparse)
    pred = mod.predict(X_sparse)
    assert pred.shape == (10,)
    assert all(pred <= 1) and all(pred >= 0)
