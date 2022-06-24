"""Class for logistic SGD."""
import scipy.sparse
import numpy as np


class Logistic_SGD:
    """Class for logistic SGD."""

    def __init__(self, chunksize=10000):
        """Initialize Logistic_SGD object."""
        self.chunksize = chunksize

    def _loss(self, p, y):
        y = y.A
        y = y.reshape(-1)
        p = p.reshape(-1)
        return (-y * np.log(p) - (1 - y) * np.log(1 - p)).mean()

    def _logit(self, w, X):
        w = scipy.sparse.csr_matrix(w).reshape(-1, 1)
        p = 1 / (1 + np.exp(-((X.dot(w)).A)))
        return p

    def _grad(self, p, y, X, learning_rate):
        score = learning_rate * X.transpose().dot(p - y)
        return np.array(score)

    def _update_weights(self, y, X, w, learning_rate):
        p = self._logit(w, X).reshape(-1, 1)
        score = self._grad(p, y, X, learning_rate)
        w = w - score
        return p, w

    def _sgd_iterative(self, X, y):
        chunksize = self.chunksize
        total_chunks = int(X.shape[0] / chunksize)
        w = np.zeros(X.shape[1]).reshape(-1, 1)
        weights_matrix = np.zeros((X.shape[1], total_chunks + 1))
        weights_matrix[:, 0] = w.reshape(-1)
        log_likelihood = np.zeros(total_chunks)
        for chunk_no in range(total_chunks):
            X_chunk = X[chunk_no * chunksize:(chunk_no + 1) * chunksize, :]
            y_chunk = y[chunk_no * chunksize:(chunk_no + 1) * chunksize, :]
            learning_rate = 1 / (np.sqrt(chunk_no + 1) * 10000)
            p, w = self._update_weights(y_chunk, X_chunk, w, learning_rate)

            # getting log_likelihood
            log_likelihood[chunk_no] = self._loss(p, y_chunk)

            # updating matrix of weights
            weights_matrix[:, chunk_no + 1] = w.reshape(-1)

        return p, w, weights_matrix, log_likelihood

    def fit(self, X, y):
        """Fit the model."""
        p, w, weights_matrix, log_likelihood = self._sgd_iterative(X, y)
        self.p = p
        self.w = w
        self.weights_matrix = weights_matrix
        self.log_likelihood = log_likelihood
        return self

    def predict(self, X):
        """Predict probabilities."""
        return self._logit(self.w, X).reshape(-1, 1)
