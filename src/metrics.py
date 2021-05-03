from typing import Iterable
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import LatentDirichletAllocation, KernelPCA
from pandas.core.frame import DataFrame
from tmtoolkit.topicmod.evaluate import metric_coherence_gensim
import pandas as pd

from sklearn.metrics import silhouette_score
import numpy as np


def inverse_score(estimator: KernelPCA, X: Iterable, y=None) -> float:
    """Inverse score is a score measure for fitting the Kernel PCA
    algorithm. This is used to learn the hyperparameters of that model

    Args:
        estimator (KernelPCA): model
        X (Iterable): Entries (vectorized tweets)
        y (Iterable, optional): (Not used). Defaults to None.

    Returns:
        float: inverse score
    """
    X_reduced = estimator.transform(X)
    X_preimage = estimator.inverse_transform(X_reduced)
    return -1 * mean_squared_error(X, X_preimage)


def lda_scorer(estimator: LatentDirichletAllocation, X: DataFrame,
               n_keywords: int, tfidf: TfidfVectorizer) -> dict:
    """Function that specifies a score function
    for the LDA algorithm hyperparameter tuning.
    It uses the coherence score and the perplexity specific
    score for the LDA

    Args:
        estimator (LatentDirichletAllocation): [description]
        X (DataFrame): Data, vectorized using TF-IDF or other technique
        n_keywords: Number of keywords to keep into account
        tfidf; vectorizer
    Returns:
        dict: Dictionary of coherence and perp
    """
    estimator.fit(X)
    dtm_tf = pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names())

    clda = metric_coherence_gensim(measure='u_mass',
                                   top_n=n_keywords,
                                   topic_word_distrib=estimator.components_,
                                   dtm=dtm_tf,
                                   vocab=np.array(
                                       [x for x in tfidf.vocabulary_.keys()],
                                       dtype=object),
                                   return_mean=True)
    return {'coherence': clda, 'perp': estimator.perplexity(X)}


def silhouette_scorer(estimator: KMeans, X: Iterable, y=None) -> float:
    """Silhouette scorer. Silhouette measures how "well"
    done are the clusters (how differenciated they are)

    Args:
        estimator (KMeans): Kmeans estimator
        X (Iterable): List of vectors or matrix of vectorized tweets
        or dimension reductioned TF-IDF tweeets
        y (Iterable, optional): Not used. Defaults to None.

    Returns:
        float: silhouette score
    """
    labs = estimator.fit_predict(X)
    return silhouette_score(X, labels=labs, metric='cosine')
