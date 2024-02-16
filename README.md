<!--- BADGES: START --->
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/ksolarski/effCTR)
[![PyPI](https://img.shields.io/pypi/v/effCTR)][#pypi-package]
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/d8576bcc9a43423883beea2c5895bac6)](https://app.codacy.com/gh/ksolarski/effCTR?utm_source=github.com&utm_medium=referral&utm_content=ksolarski/effCTR&utm_campaign=Badge_Grade)
[![Downloads](https://pepy.tech/badge/effctr)](https://pepy.tech/project/effctr)
[![Build Status](https://github.com/ksolarski/effCTR/actions/workflows/build.yml/badge.svg)](https://github.com/ksolarski/effCTR/actions)
[![codecov](https://codecov.io/gh/ksolarski/effCTR/graph/badge.svg?token=DNYKBCLNKU)](https://codecov.io/gh/ksolarski/effCTR)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/effCTR)][#pypi-package]
[![PyPI - License](https://img.shields.io/pypi/l/effCTR?logo=pypi&style=flat&color=green)][#github-license]

[#github-license]: https://github.com/ksolarski/effCTR/blob/master/LICENSE
[#pypi-package]: https://pypi.org/project/ksolarski/effCTR/
<!--- BADGES: END --->

# effCTR

Efficient Click-Through Rate (effCTR) implements Logistic Regression with Stochastic Gradient Descent (SGD) and Naive Bayes in Python, and it utilises sparse matrices from [scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html) to achive up to 60x speedup compared to implementations from [scikit-learn](https://scikit-learn.org/stable/).

Below, more information is provided about Naive Bayes and Logistic Regression with SGD developed in this repository. For each algorithm, a simple user guide is presented. Furthermore, a notebook with an example of how algorithms can be used on the real dataset can be found [here](https://github.com/kapa112/effCTR/blob/main/notebooks/demo.ipynb).

## Installation

To install the package from PyPI:

```bash
pip3 install effCTR
```

One can also install it directly from github:

```bash
pip3 install git+https://github.com/ksolarski/effCTR.git
```

## Logistic Regression with SGD

Due to the high dimension of the matrix in this problem, Logistic Regression is infeasible since the matrix cannot be inverted. Hence, Stochastic Gradient Descent is applied to this problem. The loss function is specified as:

$$
\begin{equation}
Log Loss =-\left[y_{t} \log \left(p_{t}\right)+\left(1-y_{t}\right) \log \left(1-p_{t}\right)\right]
\end{equation}
$$

Then the gradient is given by:

$$
\begin{equation}
\nabla Log Loss=\left(p_{t}-y_{t}\right) x_{t}
\end{equation}
$$

Hence, in each iteration of the algorithm, the weights are updated in the following way:

$$
\begin{equation}
w_{t+1} = w_{t}-\eta_{t}\left(p_{t}-y_{t}\right) x_{t}
\end{equation}
$$

where $\eta_{t}$ denotes learning rate at iteration $t$, which can be specified in the argument ``learning_rate``. One can also specify how many times to go through a dataset in the argument ``max_epochs``, how large data chunk is used in each iteration in the argument ``chunksize``, and whether to iterate through consecutive batches in dataset or draw a batch randomnly in the argumnet ``randomized``. One can also add early stopping by using arguments ``early_stopping`` and ``n_iter_no_change``. **The [demo](https://github.com/kapa112/effCTR/blob/main/notebooks/demo.ipynb) shows that the implementation from this repo can be ~60x faster on large datasets than the implementation from [scikit-learn](https://scikit-learn.org/stable/modules/sgd.html)**.

To fit the model:

```python3
from effCTR.models.Logistic_SGD import Logistic_SGD
Logistic_SGD = Logistic_SGD()
Logistic_SGD.fit(X, y)
```

Afterwards, one can otain predictions:

```python3
Logistic_SGD = Logistic_SGD.predict(X)
```

One can also use methods ``plot_weights`` and ``log_likelihood`` to plot how weights and likelihood change throughout the training process.

## Naive Bayes

Under assumption of conditional independence and using Bayes theorem, one can show that probability of click given set of features $X$ can be obtained using:

$$
\begin{equation}
P(Y=1 \mid X=x) =\frac{P(Y=1) \prod_{i} P\left(X_{i}=x_{i} \mid Y=1\right)}{P(Y=1) \prod_{i} P\left(X_{i}=x_{i} \mid Y=1\right) + P(Y=0) \prod_{i} P\left(X_{i}=x_{i} \mid Y=0\right)}
\end{equation}
$$

where $P\left(X_{i}=x_{i} \mid Y=1\right)$ denotes probabilty of feature $X_i$ taking a value $x_i$ given that there is a click. Using logarithms, one obtains alternative expression, which enables us to utilize fast operations from [scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html):

$$
\begin{equation}
\begin{aligned}
P(Y=1) \prod_{i} P\left(X_{i}=x_{i} \mid Y=1\right) = \exp [ {\log \{{P(Y=1) \prod_{i} P\left(X_{i}=x_{i} \mid Y=1\right)} ] }} = \\
\exp [ {\log{P(Y=1)} + \sum_{i} \log{P\left(X_{i}=x_{i} \mid Y=1\right)}} ]
\end{aligned}
\end{equation}
$$

However, estimated probability can be zero (no observations for particular feature vaue given click or given no click). Consequently, zero probabilities are replaced either by small $\epsilon$ or smallest positive values encountered in the dataset. User can specify this using the arguments ``replace_by_epsilon`` and ``epsilon``.

To fit the model:

```python3
from effCTR.models.Naive_Bayes import Naive_Bayes
Naive_Bayes = Naive_Bayes()
Naive_Bayes.fit(X, y)
```

Afterwards, one can otain predictions:

```python3
preds_bayes = Naive_Bayes.predict(X)
```

## Contributing & Future Work

There is no clear path for this project. It was created for fun and learning purposes. Contributions and collaborations are welcomed and encouraged.
