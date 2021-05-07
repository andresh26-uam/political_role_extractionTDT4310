from typing import List, Tuple
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import KernelPCA
from src import INV_SCORE, LAST_SCORES_ROUTE,\
    SCORE_NAMES, SILHOUETTE, TRAINED_CLUSTERER_ROUTE
from src.metrics import inverse_score, silhouette_scorer
from sklearn.model_selection import GridSearchCV

from src.utils import print_best_params, showclusters
import joblib

TRAIN_PARAM_GRID_CLUSTERER = {
    "preprocessor__gamma": [0.03, 0.05, 0.1],
    "preprocessor__n_components": [6, 8, 12, 20],
    "preprocessor__kernel": ["cosine", "rbf", "linear"],
    "kmeans__max_iter": [100, ],
    "kmeans__n_init": [50],
    "kmeans__n_clusters": [3, 4, 5, 6]
}


def clusterize(all_data: List[dict], all_data_t: List[List[float]],
               retrain=False,
               param_grid=TRAIN_PARAM_GRID_CLUSTERER,
               plot=False,
               random_st=42) -> Tuple[BaseEstimator, pd.DataFrame, List[List]]:
    """Clusterizes the input tweets, outputing them in original form with
    cluster label (see added 'predicted_cluster' key) and the clusters
    in matrix form, where each row corresponds to the assigned probability
    distribution of the cluster labels where the tweet is classified).
    First argument, on the first hand, is
    the newly trained (or read, depending on "retrain" argument supplied)
    clustering model, which uses KPCA and Kmeans. You can change the training
    parameters in __init__.py on this folder. Also, if you run
    $ python train.py -b
    the best estimators will be used in the 'param_grid' argument instead of
    the default training parameters (TRAIN_PARAM_GRID_CLUSTERER in __init__.py)

    Args:
        all_data (List[dict]): Cleaned tweets in dict format
        all_data_t (List[List[float]]): Vectorization of tweets
        retrain (bool, optional): If True, trains a newly created
        model with the GridSearch parameters told in 'param_grid'.
            Otherwise, a model is read from TRAINED_CLUSTERER_ROUTE
            (found in __init__.py as well). Defaults to False.
        param_grid (dict, optional): Dictionary of hyperparameters to tune
            in the training process. Ignored when 'retrain==False'.
            Defaults to TRAIN_PARAM_GRID_CLUSTERER.
        plot (bool, optional): Whether to plot the clusters
            after training/reading the model. Defaults to False.
        random_st (int, optional): Random seed. Defaults to 42.

    Returns:
        Tuple[BaseEstimator, pd.DataFrame, List[List]]: clusterer trained,
            dataframe with original tweets with 'predicted_cluster' label,
            vector of prob. distribution of tweets over labels.
    """
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
    with open(LAST_SCORES_ROUTE, "a+") as f:
        print(SCORE_NAMES[INV_SCORE], file=f)
        print(str(inverse_score(clusterer['preprocessor'],
                                all_data_t.toarray())), file=f)
        print(SCORE_NAMES[SILHOUETTE], file=f)
        print(str(silhouette_scorer(
            clusterer['kmeans'],
            clusterer['preprocessor'].transform(
                all_data_t.toarray()))), file=f)
    return clusterer, data_preprocessed_df, clusters
