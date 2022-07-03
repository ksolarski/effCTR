"""Class for Naive Bayes."""
import numpy as np


class Naive_Bayes:
    """Class for Naive Bayes."""

    def __init__(self, replace_by_epsilon=True, epsilon=1e-5):
        """Initialize Naive_Bayes object."""
        self.replace_by_epsilon = replace_by_epsilon
        self.epsilon = epsilon

    def _replace_0_p_epsilon(self, p_given_click,
                             p_given_no_click, epsilon):
        "Replace 0 probabilities with epsilon."
        p_given_click[p_given_click == 0] = epsilon
        p_given_no_click[p_given_no_click == 0] = epsilon

        return p_given_click, p_given_no_click

    def _replace_0_p_sample(self, p_given_click, p_given_no_click):
        "Replace 0 probabilities with smalles observed probability."
        min_p_given_click = np.min(
            p_given_click[p_given_click > 0])
        p_given_click[p_given_click == 0] = min_p_given_click

        min_p_given__no_click = np.min(
            p_given_no_click[p_given_no_click > 0])
        p_given_no_click[p_given_no_click == 0] = min_p_given__no_click

        return p_given_click, p_given_no_click

    def fit(self, X, y, epsilon=None):
        """Obtain probabilities of features given click and given no click.

        Parameters
        ----------
        X : sparse matrix with shape (n_samples, n_features)
            Training data.
        y : sparse matrix with shape (n_samples, 1)
            Target values.
        epsilon : float.
            Small number used to replace 0 probabilities

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        # getting probability of click and no click
        self.p_click = y.mean()

        # getting the indices of rows correponding to click and not click
        click_indices = y.A.reshape(-1)
        no_click_indices = 1 - click_indices
        list_of_indices_click = np.array((np.where(click_indices))).tolist()[0]
        list_of_indices_no_click = np.array(
            (np.where(no_click_indices))).tolist()[0]

        # getting matrices of features corresponding
        # to observations with click and no click
        m_clicks = X[list_of_indices_click, :]
        m_no_clicks = X[list_of_indices_no_click, :]

        # getting probabilities for each set
        # of features given click and given no click
        p_given_click = np.array(
            m_clicks.mean(axis=0).tolist()[0])
        p_given_no_click = np.array(
            m_no_clicks.mean(axis=0).tolist()[0])

        # replacing zero probabilities
        if self.replace_by_epsilon is True:
            if epsilon is None:
                epsilon = self.epsilon
            p_given_click, p_given_no_click = self._replace_0_p_epsilon(
                p_given_click, p_given_no_click, epsilon)
        else:
            p_given_click, p_given_no_click = self._replace_0_p_sample(
                p_given_click, p_given_no_click)

        self.p_given_click = p_given_click
        self.p_given_no_click = p_given_no_click

        return self

    def predict(self, X):
        """Obtain predictions.

        Parameters
        ----------
        X : sparse matrix with shape (n_samples, n_features)
            Data used to obtain predictions.

        Returns
        -------
        np.array of size (n_samples,) with predictions.
        """
        # getting logs of probabilities
        p_logs_given_click = np.log(self.p_given_click)
        p_logs_given_no_click = np.log(self.p_given_no_click)

        # summing probabilities - using dot product
        sum_p_logs_given_click = X.dot(
            p_logs_given_click)
        sum_p_logs_given_no_click = X.dot(
            p_logs_given_no_click)

        # getting logs of probabilities of click and no click
        p_click = self.p_click
        log_p_click = np.log(self.p_click)
        log_p_no_click = np.log((1 - p_click))

        # calculating numerator and denominator separately
        log_p_num = sum_p_logs_given_click + log_p_click
        log_p_den = sum_p_logs_given_no_click + log_p_no_click
        numerator = np.exp(log_p_num)
        denominator = numerator + np.exp(log_p_den)
        preds = numerator / denominator

        return preds
