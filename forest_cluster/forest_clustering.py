from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division

from joblib import Parallel, delayed
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_random_state


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


def feature_similarity(leaf_one, leaf_two):
    return np.sum(leaf_one == leaf_two) / leaf_one.shape[0]


class RandomForestClusterer(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self, bootstrap=True, compute_importances=None,
                 criterion='gini', max_depth=None, max_features='auto',
                 max_leaf_nodes=None, min_density=None, min_samples_leaf=1,
                 min_samples_split=2, n_estimators=10, n_jobs=1,
                 oob_score=False, random_state=None, verbose=0):
        self.bootstrap = bootstrap
        self.compute_importances = compute_importances
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.oob_score = oob_score
        self.random_state = random_state
        self.verbose = verbose

        self.estimator_ = None


    def  _init_estimator(self):
        return RandomForestClassifier(
                bootstrap=self.bootstrap,
                #compute_importances=self.compute_importances,
                criterion=self.criterion,
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                n_estimators=self.n_estimators,
                n_jobs=self.n_jobs,
                oob_score=self.oob_score,
                random_state=self.random_state,
                verbose=self.verbose)

    def build_similarity_matrix(self, X):
        n_samples, n_features = X.shape
        leaves = self.estimator_.apply(X)
        similarity = np.zeros((n_samples, n_samples))
        triu = np.triu_indices_from(similarity)
        indices = zip(list(triu[0]), list(triu[1]))

        results = Parallel(n_jobs=8)(
                delayed(feature_similarity)(leaves[i1, :], leaves[i2, :])
                for i1, i2 in indices)

        similarity[triu] = np.array(results)
        similarity += (similarity.T - np.diag(np.diag(similarity)))

        return similarity


    def fit(self, X, y=None):
        self.estimator_ = self._init_estimator()
        X_, y_ = generate_dataset(X)

        self.estimator_.fit(X_, y_)

        return self

    def fit_predict(self, X, y=None):
        pass
