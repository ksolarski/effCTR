"""Class for logistic SGD."""
import scipy.sparse
import numpy as np


def decaying_learning_rate(t):
    """Decaying learning rate."""
    return 1 / (np.sqrt(t + 1) * 10000)


class Logistic_SGD:
    """Class for logistic SGD."""

    def __init__(self, chunksize=10000,
                 learning_rate=decaying_learning_rate,
                 max_epochs=10,
                 randomized=False,
                 early_stopping=False,
                 n_iter_no_change=50):
        """Initialize Logistic_SGD object."""
        self.chunksize = chunksize
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.randomized = randomized
        self.early_stopping = early_stopping
        self.n_iter_no_chang = n_iter_no_change

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

        # initialize
        chunksize = self.chunksize
        total_chunks = int(X.shape[0] / chunksize)
        w = np.zeros(X.shape[1]).reshape(-1, 1)
        weights_matrix = []
        log_likelihood = []
        for epoch in range(self.max_epochs):
            for chunk_no in range(total_chunks):
                if self.randomized:
                    chunk_no = np.random.randint(0, total_chunks)
                X_chunk = X[chunk_no * chunksize:(chunk_no + 1) * chunksize, :]
                y_chunk = y[chunk_no * chunksize:(chunk_no + 1) * chunksize, :]
                learning_rate = self.learning_rate(chunk_no)
                p, w = self._update_weights(y_chunk, X_chunk, w, learning_rate)

                # getting log_likelihood
                log_likelihood.append(self._loss(p, y_chunk))

                # implement early stopping
                if self.early_stopping is True:
                    min_stop = min(len(log_likelihood), self.n_iter_no_chang)
                    if log_likelihood[-1] > (log_likelihood[-min_stop]):
                        return w, weights_matrix, log_likelihood

                # updating matrix of weights
                weights_matrix.append(w.reshape(-1))

        return w, weights_matrix, log_likelihood

    def fit(self, X, y):
        """Fit the model."""
        w, weights_matrix, log_likelihood = self._sgd_iterative(X, y)
        self.w = w
        self.weights_matrix = weights_matrix
        self.log_likelihood = log_likelihood
        return self

    def predict(self, X):
        """Predict probabilities."""
        return self._logit(self.w, X).reshape(-1, 1)
