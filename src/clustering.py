from typing import List, Tuple
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import KernelPCA
from src import TRAINED_CLUSTERER_ROUTE
from src.metrics import inverse_score, silhouette_scorer
from sklearn.model_selection import GridSearchCV

from src.utils import print_best_params, showclusters
import joblib

TRAIN_PARAM_GRID_CLUSTERER = {
    "preprocessor__gamma": [0.03, 0.05, 0.1],
    "preprocessor__n_components": [4, 8, 16, 24],
    "preprocessor__kernel": ["cosine", "linear"],
    "kmeans__max_iter": [100, ],
    "kmeans__n_init": [20, 50],
    "kmeans__n_clusters": [2, 3, 4, 6]
}


def clusterize(all_data: List[dict], all_data_t: List[List[float]],
               retrain=False,
               param_grid=TRAIN_PARAM_GRID_CLUSTERER,
               plot=False,
               random_st=42) -> Tuple[BaseEstimator, pd.DataFrame, List[List]]:

    if retrain:

        # https://stackoverflow.com/questions/53556359/selecting-kernel-and-hyperparameters-for-kernel-pca-reduction
        clusterer = Pipeline(
            [
                (
                    "preprocessor",
                    KernelPCA(
                        random_state=random_st,
                        fit_inverse_transform=True, n_jobs=-1
                    ),
                ),
                (
                    "kmeans",
                    KMeans(
                        init="k-means++",
                        random_state=random_st
                    ),
                ),
            ]
        )
        gsearch = GridSearchCV(
            clusterer, param_grid, cv=3,
            scoring=lambda est, X:
                {'silhouette': silhouette_scorer(
                    est['kmeans'],
                    est['preprocessor'].fit_transform(X)),
                    'inv_score': inverse_score(est['preprocessor'], X)},
            refit='silhouette', n_jobs=-1)

        clusterer = gsearch.fit(all_data_t.toarray())
        print_best_params(gsearch, "KPCA and Kmeans")
        clusterer = clusterer.best_estimator_
        joblib.dump(clusterer, TRAINED_CLUSTERER_ROUTE)
    else:
        clusterer = joblib.load(TRAINED_CLUSTERER_ROUTE)

    KPCA_best = clusterer['preprocessor']
    Kmeans_best = clusterer['kmeans']
    data_preprocessed = KPCA_best.transform(
        all_data_t.toarray())

    data_preprocessed_df = pd.DataFrame(
        data_preprocessed,
        columns=["component_{}".format(str(i)) for i in range(
            0, KPCA_best.get_params()['n_components'])]
    )

    clusters = Kmeans_best.fit_predict(data_preprocessed_df)
    data_preprocessed_df["predicted_cluster"] = Kmeans_best.labels_
    data_preprocessed_df["tweets"] = all_data
    if plot:
        showclusters(data_preprocessed_df)
    return clusterer, data_preprocessed_df, clusters