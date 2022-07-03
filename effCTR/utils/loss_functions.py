"""Loss functions used by models."""
import numpy as np


def log_loss(p, y):
    "Obtain log loss."
    y = y.A
    y = y.reshape(-1)
    p = p.reshape(-1)
    return (-y * np.log(p) - (1 - y) * np.log(1 - p)).mean()
