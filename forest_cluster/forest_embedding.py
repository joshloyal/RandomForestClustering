from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble.forest import BaseForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state, check_array
from sklearn.preprocessing import OneHotEncoder


def bootstrap_sample_column(X, n_samples=None, random_state=1234):
    """bootstrap_sample_column

    Bootstrap sample the column of a dataset.

    Parameters
    ----------
    X : np.ndarray (n_samples,)
        Column to bootstrap

    n_samples : int
        Number of samples to generate. If `None` then generate
        a bootstrap of size of `X`.

    random_state : int
        Seed to the random number generator.

    Returns
    -------
    np.ndarray (n_samples,):
        The bootstrapped column.
    """
    random_state = check_random_state(random_state)
    if n_samples is None:
        n_samples = X.shape[0]

    return random_state.choice(X, size=n_samples, replace=True)


def uniform_sample_column(X, n_samples=None, random_state=1234):
    """uniform_sample_column

    Sample a column uniformly between its minimum and maximum value.

    Parameters
    ----------
    X : np.ndarray (n_samples,)
        Column to sample.

    n_samples : int
        Number of samples to generate. If `None` then generate
        a bootstrap of size of `X`.

    random_state : int
        Seed to the random number generator.

    Returns
    -------
    np.ndarray (n_samples,):
        Uniformly sampled column.
    """
    random_state = check_random_state(random_state)
    if n_samples is None:
        n_samples = X.shape[0]

    min_X, max_X = np.min(X), np.max(X)
    return random_state.uniform(min_X, max_X, size=n_samples)


def generate_synthetic_features(X, method='bootstrap', random_state=1234):
    """generate_synthetic_features

    Generate a synthetic dataset based on the empirical distribution
    of `X`.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_features)
        Dataset whose empirical distribution is used to generate the
        synthetic dataset.

    method : str {'bootstrap', 'uniform'}
        Method to use to generate the synthetic dataset. `bootstrap`
        samples each column with replacement. `uniform` generates
        a new column uniformly sampled between the minimum and
        maximum value of each column.

    random_state : int
        Seed to the random number generator.

    Returns
    -------
    synth_X : np.ndarray (n_samples, n_features)
        The synthetic dataset.
    """
    random_state = check_random_state(random_state)
    n_features = int(X.shape[1])
    synth_X = np.empty_like(X)
    for column in xrange(n_features):
        if method == 'bootstrap':
            synth_X[:, column] = bootstrap_sample_column(
                X[:, column], random_state=random_state)
        elif method == 'uniform':
            synth_X[:, column] = uniform_sample_column(
                X[:, column], random_state=random_state)
        else:
            raise ValueError('method must be either `bootstrap` or `uniform`.')

    return synth_X


def generate_discriminative_dataset(X, method='bootstrap', random_state=1234):
    """generate_discriminative_dataset.

    Generate a synthetic dataset based on the empirical distribution
    of `X`. A target column will be returned that is 0 if the row is
    from the real distribution, and 1 if the row is synthetic. The
    number of synthetic rows generated is equal to the number of rows
    in the original dataset.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_features)
        Dataset whose empirical distribution is used to generate the
        synthetic dataset.

    method : str {'bootstrap', 'uniform'}
        Method to use to generate the synthetic dataset. `bootstrap`
        samples each column with replacement. `uniform` generates
        a new column uniformly sampled between the minimum and
        maximum value of each column.

    random_state : int
        Seed to the random number generator.

    Returns
    -------
    X_ : np.ndarray (2 * n_samples, n_features)
        Feature array for the synthetic dataset. The rows
        are randomly shuffled, so synthetic and actual samples should
        be intermixed.

    y_ : np.ndarray (2 * n_samples)
        Target column indicating whether the row is from the actual
        dataset (0) or synthetic (1).
    """
    random_state = check_random_state(random_state)
    n_samples = int(X.shape[0])

    synth_X = generate_synthetic_features(
        X, method=method, random_state=random_state)
    X_ = np.vstack((X, synth_X))
    y_ = np.concatenate((np.ones(n_samples), np.zeros(n_samples)))

    permutation_indices = random_state.permutation(np.arange(X_.shape[0]))
    X_ = X_[permutation_indices, :]
    y_ = y_[permutation_indices]

    return X_, y_


class RandomForestEmbedding(BaseForest):
    """Very similar to sklearn's RandomTreesEmbedding;
    however, the forest is trained as a discriminator.
    """
    def __init__(self,
                 n_estimators=10,
                 criterion='gini',
                 max_depth=5,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features='auto',
                 max_leaf_nodes=None,
                 bootstrap=True,
                 sparse_output=True,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(RandomForestEmbedding, self).__init__(
                base_estimator=DecisionTreeClassifier(),
                n_estimators=n_estimators,
                estimator_params=("criterion", "max_depth", "min_samples_split",
                                  "min_samples_leaf", "min_weight_fraction_leaf",
                                  "max_features", "max_leaf_nodes",
                                  "random_state"),
                bootstrap=bootstrap,
                oob_score=False,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.sparse_output = sparse_output

    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported in tree embedding")

    def fit(self, X, y=None, sample_weight=None):
        self.fit_transform(X, y, sample_weight=sample_weight)
        return self

    def fit_transform(self, X, y=None, sample_weight=None):
        X = check_array(X, accept_sparse=['csc'], ensure_2d=False)

        if sp.issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        X_, y_ = generate_discriminative_dataset(X)

        super(RandomForestEmbedding, self).fit(X_, y_,
                                               sample_weight=sample_weight)

        self.one_hot_encoder_ = OneHotEncoder(sparse=True)
        if self.sparse_output:
            return self.one_hot_encoder_.fit_transform(self.apply(X))
        return self.apply(X)

    def transform(self, X):
        if self.sparse_output:
            return self.one_hot_encoder_.fit_transform(self.apply(X))
        return self.apply(X)
