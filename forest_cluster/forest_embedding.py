from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division

from joblib import Parallel, delayed
import numpy as np
from sklearn.ensemble.forest import BaseForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state, check_array
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sp


def generate_synthetic(X, random_state=1234):
    random_state = check_random_state(random_state)
    n_samples, n_features = X.shape
    synth_X = np.empty_like(X)
    for column in xrange(int(n_features)):
        synth_X[:, column] = random_state.choice(X[:, column], size=int(n_samples), replace=True)
    return synth_X


def generate_dataset(X, random_state=1234):
    random_state = check_random_state(random_state)
    n_samples = int(X.shape[0])

    synth_X = generate_synthetic(X)
    X_ = np.vstack((X, synth_X))
    y_ = np.concatenate((np.ones(n_samples), np.zeros(n_samples)))

    permutation_indices = random_state.permutation(np.arange(X_.shape[0]))
    X_ = X_[permutation_indices, :]
    y_ = y_[permutation_indices]

    return X_, y_


def feature_similarity_single(leaf_one, leaf_two):
    return np.sum(leaf_one == leaf_two) / leaf_one.shape[0]


def random_forest_dissimilarity_single(leaf_one, leaf_two):
    return 1. - feature_similarity_single(leaf_one, leaf_two)


def random_forest_dissimilarity(X):
    n_samples, n_features = X.shape
    similarity = np.zeros((n_samples, n_samples))
    triu = np.triu_indices_from(similarity)
    indices = zip(list(triu[0]), list(triu[1]))

    results = Parallel(n_jobs=8)(
            delayed(hamming)(X[i1, :], X[i2, :])
            for i1, i2 in indices)

    similarity[triu] = np.array(results)
    similarity += (similarity.T - np.diag(np.diag(similarity)))

    return similarity


class RandomTreesEmbedding(BaseForest):
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
        super(RandomTreesEmbedding, self).__init__(
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

        X_, y_ = generate_dataset(X)

        super(RandomTreesEmbedding, self).fit(X_, y_,
                                              sample_weight=sample_weight)

        self.one_hot_encoder_ = OneHotEncoder(sparse=True)
        if self.sparse_output:
            return self.one_hot_encoder_.fit_transform(self.apply(X))
        return self.apply(X)

    def transform(self, X):
        if self.sparse_output:
            return self.one_hot_encoder_.fit_transform(self.apply(X))
        return self.apply(X)
