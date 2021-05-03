from typing import Any, List, Tuple
import joblib
from sklearn.base import BaseEstimator

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from src import LDA_MAXITER, TRAINED_LDA_ROUTE
from src.metrics import lda_scorer
from src.utils import plot_top_words, print_best_params
TRAIN_PARAM_GRID_LDA = {'n_components': [10, 15, 20, 30],
                        'doc_topic_prior':
                            [0.05, 0.1, 0.2, 0.33, 0.5, 0.75, 0.95],
                        'topic_word_prior':
                            [0.05, 0.1, 0.2, 0.33, 0.5, 0.75, 0.95]}


def topic_modelling(all_data_t: List[List[float]],
                    tfidf: TfidfVectorizer, n_keywords: int,
                    param_grid=TRAIN_PARAM_GRID_LDA,
                    retrain=False, plot=False,
                    random_seed=42) -> Tuple[BaseEstimator,
                                             List[List[float]], Any]:
    """Performs the topic modelling task over a set of vectorized tweets.
    The results are:
    - Trained LDA model
    - Transform of the model (tweet-topic distribution)
    - Topics of the model (topic-word distribution)
    If 'retrain' is False, then, no training is performed and the model
    is retrieved from TRAINED_LDA_ROUTE (found in __init__.py).
    If the command is:
    $ python train.py -b
    'param_grid' will be set to the best found combination of parameters
    from previous execution of train.py (this file is overriden each time
    'train.py' is executed).

    Args:
        all_data_t (List[List[float]]): Vectorized tweets
        tfidf (TfidfVectorizer): TF-IDF vectorizer
        n_keywords (int): Number of keywords used to plot the topics
            (ignored if 'plot' is True)
        param_grid ([type], optional): Parameters to try to tune.
            Each key is a hyperparameter of the
            LatentDirichletAllocation class, and the values are
            user-specified worth trying possible values.
            Defaults to TRAIN_PARAM_GRID_LDA. (This is similar
            in clustering.py)
        retrain (bool, optional): If true, retrains the LDA model
            optimizing the 'param_grid' possibilities
            Otherwise, tries to fetch the previously trained model.
            Defaults to False.
        plot (bool, optional): Plots the results. Defaults to False.
        random_seed (int, optional): Random seed. Defaults to 42.

    Returns:
        Tuple[BaseEstimator, List[List[float]], Any]: Result is:
            - Trained LDA model
            - Transform of the model (tweet-topic distribution)
            - Topics of the model (topic-word distribution)
    """
    param_args = param_grid
    tf_feature_names = tfidf.get_feature_names()
    if retrain:
        model = GridSearchCV(
            LatentDirichletAllocation(max_iter=LDA_MAXITER,
                                      random_state=random_seed),
            param_grid=param_args,
            scoring=lambda est, X: lda_scorer(
                est, X, n_keywords, tfidf),
            refit='coherence')
        r = model.fit(all_data_t)
        print_best_params(r, "LDA model")
        lda_model = r.best_estimator_
        joblib.dump(lda_model, TRAINED_LDA_ROUTE)
    else:
        lda_model = joblib.load(TRAINED_LDA_ROUTE)

    n_top_words = n_keywords
    print("Example of a tweet-topic distribution")
    print(str(lda_model.transform(all_data_t[0])))
    if plot:
        print("Plotting topics... (close window to continue)")
        plot_top_words(lda_model, tf_feature_names,
                       n_top_words, 'Topics in LDA model')
        print("Done.")

    topic_word_dist = lda_model.components_
    # Clusterize into classes
    # Matrix of documents (rows) with its topic probability (columns):
    all_data_tocluster = lda_model.transform(all_data_t)
    return lda_model, all_data_tocluster, topic_word_dist
