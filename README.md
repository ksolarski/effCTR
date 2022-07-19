# Background

Part of this project was done in a group of 3 students during my master studies at the University of Amsterdam. It implements Naive Bayes and Logistic Regression with Stochastic Gradient Descent (SGD) for click-through rate competition from Kaggle: [Click-Through Rate Prediction](https://www.kaggle.com/c/avazu-ctr-prediction/overview). The focus in this project is on implementing both algorithms from scratch (without using any ML packages) and using sparse matrices from [scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html) for speed and efficiency.

Below more information is provided about Naive Bayes and Logistic Regression with SGD developed in this repository. For each algorithm, simple user guide is presented. Furthermore, a notebook with an example how algorithms can be used on real dataset can be found [here](https://github.com/kapa112/effCTR/blob/main/notebooks/Avazu_CTR_Predictions.ipynb).

# Installation

The package is not officaily published through pypi, however it can be installed directly from github:
```bash
pip3 install git+https://github.com/kapa112/effCTR.git
```

# Naive Bayes

### Algorithm Explanation

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

### Usage

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

# Logistic Regression with SGD

### Algorithm Explanation

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

where $\eta_{t}$ denotes learning rate at iteration $t$, which can be specified in the argument ``learning_rate``. One can also specify how many times to go through a dataset in the argument ``max_epochs``, how large data chunk is used in each iteration in the argument ``chunksize``, and whether to iterate through consecutive batches in dataset or draw a batch randomnly in the argumnet ``randomized``. One can also add early stopping by using arguments ``early_stopping`` and ``n_iter_no_change``.

### Usage

Same as with Naive Bayes:

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

# Contributing & Future Work

This project was done for learning purposes and is not maintained. However, if you have an idea about the new path for this project, please comment and contribute :)
