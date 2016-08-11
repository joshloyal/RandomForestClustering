from sklearn.manifold import MDS
import forest_cluster as rfc
from forest_cluster.tests.fixtures import generate_clustered_data
import numpy as np
from mysuper.datasets import fetch_cars, fetch_10kdiabetes

import seaborn as sns
import pandas as pd

#X = fetch_cars().values
X = fetch_10kdiabetes().values[: 5000]

cluster = rfc.RandomForestClusterer(n_estimators=20)
cluster.fit(X)

print('building similarity matrix')
Z = cluster.build_similarity_matrix(X)
Z = np.sqrt(1 - Z)

print('embedding data')
projector = MDS(dissimilarity='precomputed')
embedding = projector.fit_transform(Z)

df = pd.DataFrame({'x': embedding[:, 0], 'y': embedding[:, 1]})
sns.jointplot('x', 'y', data=df)
