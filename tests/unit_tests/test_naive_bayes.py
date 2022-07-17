"""Test naive bayes."""
import pytest
import scipy.sparse
import numpy as np
from effCTR.models.Naive_Bayes import Naive_Bayes


@pytest.fixture(scope='function')
def y_sparse():
    """Create a sample dataset."""
    y = np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0]).reshape(-1, 1)
    y = scipy.sparse.csr_matrix(y)
    return y


@pytest.fixture(scope='function')
def X_sparse():
    """Create a sample dataset."""
    X = np.array([0, 1, 0, 1, 1, 0, 1, 1, 1, 0,
                  1, 1, 1, 0, 1, 0, 1, 0, 0, 1]).reshape(10, 2)
    X = scipy.sparse.csr_matrix(X)
    return X


def test_simple_naive_bayes(X_sparse, y_sparse):
    """Test naive bayes on a very simple case."""
    naive_bayes = Naive_Bayes()
    mod = naive_bayes.fit(X_sparse, y_sparse)
    pred = mod.predict(X_sparse)
    assert pred.shape == (10,)
    assert all(pred <= 1) and all(pred >= 0)
