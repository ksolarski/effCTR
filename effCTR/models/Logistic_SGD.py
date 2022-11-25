"""Class for logistic SGD."""
import scipy.sparse
import numpy as np
import matplotlib.pyplot as plt
from effCTR.utils.loss_functions import log_loss


class Logistic_SGD:
    """Class for logistic SGD."""

    def __init__(self,
                 chunksize=None,
                 learning_rate=0.0001,
                 max_epochs=10,
                 randomized=False,
                 early_stopping=False,
                 n_iter_no_change=50):
        """Initialize Logistic_SGD object.

        Parameters
        ----------
        chunksize : int, default None
            Number of observations used in one iteration of the algorithm.
            If None, then whole dataset is used in each iteration.
        learning_rate : float, default 0.0001
            Learning rate.
        max_epochs : int, default 10
            Maximum number of iterations through whole dataset.
        randomized : bool, default False
            Whether to iterate through consecutive batches in dataset
            or draw a batch randomnly.
        early_stopping : bool, default False
            Whether to stop algorithm if there is no improvement
            in the loss function.
        n_iter_no_change : int, default 50
            Number of iterations used in early stopping criterion.
        """
        self.chunksize = chunksize
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.randomized = randomized
        self.early_stopping = early_stopping
        self.n_iter_no_chang = n_iter_no_change

    def _logit(self, w, X):
        w = scipy.sparse.csr_matrix(w)
        p = 1 / (1 + np.exp(-((X.dot(w)).A)))
        return p

    def _grad(self, p, y, X, learning_rate):
        score = learning_rate * X.transpose().dot(p - y)
        return np.array(score)

    def _update_weights(self, y, X, w, learning_rate):
        p = self._logit(w, X)
        score = self._grad(p, y, X, learning_rate)
        w = w - score
        return p, w

    def _process_return_results(self, w, weights_matrix, log_likelihood):
        return w, np.array(weights_matrix), np.array(log_likelihood)

    def _sgd_iterative(self, X, y):
        # initialize
        chunksize = self.chunksize
        learning_rate = self.learning_rate
        w = np.zeros((X.shape[1], 1))
        weights_matrix = []
        log_likelihood = []

        # get chunksize
        chunksize = self.chunksize
        if isinstance(chunksize, int):
            total_chunks = int(X.shape[0] / chunksize)
        else:
            chunksize = X.shape[0]
            total_chunks = 1

        # iterate through epochs and chunks
        for epoch in range(self.max_epochs):
            for chunk_no in range(total_chunks):

                # randomize chunk
                if self.randomized:
                    chunk_no = np.random.randint(0, total_chunks)

                # slice X and y
                X_chunk = X[chunk_no * chunksize:(chunk_no + 1) * chunksize, :]
                y_chunk = y[chunk_no * chunksize:(chunk_no + 1) * chunksize, :]

                # update weights
                p, w = self._update_weights(y_chunk, X_chunk, w, learning_rate)

                # get log_likelihood
                log_likelihood.append(log_loss(p, y_chunk))

                # implement early stopping
                if self.early_stopping:
                    min_stop = min(len(log_likelihood), self.n_iter_no_chang)
                    if log_likelihood[-1] > (log_likelihood[-min_stop]):
                        return self._process_return_results(
                            w, weights_matrix, log_likelihood)

                # update matrix of weights
                weights_matrix.append(w.reshape(-1))

        return self._process_return_results(w, weights_matrix, log_likelihood)

    def fit(self, X, y):
        """Fit the model.

        Parameters
        ----------
        X : sparse matrix with shape (n_samples, n_features)
            Training data.
        y : sparse matrix with shape (n_samples, 1)
            Target values.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        w, weights_matrix, log_likelihood = self._sgd_iterative(X, y)
        self.w = w
        self.weights_matrix = weights_matrix
        self.log_likelihood = log_likelihood
        return self

    def predict(self, X):
        """Predict probabilities.

        Parameters
        ----------
        X : sparse matrix with shape (n_samples, n_features)
            Data used to obtain predictions.

        Returns
        -------
        np.array of size (n_samples,) with predictions.
        """
        return self._logit(self.w, X).reshape(-1)

    def plot_weights(self, indices_weights, weight_names=None):
        """Plot the updates of weights over iterations.

        Parameters
        ----------
        indices_weights : list
            Data used to obtain predictions.
        weight_names : list, default None
            List of feature names to be displayed.
            If None, then ``indices_weights`` are displayed.
        """
        if weight_names is None:
            weight_names = indices_weights
        plt.plot(self.weights_matrix[:, indices_weights])
        plt.legend(weight_names)
        plt.ylabel('weight')
        plt.xlabel('iteration')
        plt.title('Weights of selected features after each iteration')
