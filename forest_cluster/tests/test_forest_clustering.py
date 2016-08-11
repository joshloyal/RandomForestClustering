import forest_cluster as rfc
from forest_cluster.tests.fixtures import simple_cluster


def test_forest_clusterer(simple_cluster):
    X = simple_cluster()
    cluster = rfc.RandomForestClusterer()
    cluster.fit(X)
