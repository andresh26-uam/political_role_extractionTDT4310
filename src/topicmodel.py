from typing import Any, List, Tuple
import joblib
from sklearn.base import BaseEstimator

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from src import LDA_MAXITER, TRAINED_LDA_ROUTE
from src.metrics import lda_scorer
from src.utils import plot_top_words, print_best_params
TRAIN_PARAM_GRID_LDA = {'n_components': [10, 15, 20],
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
        plot_top_words(lda_model, tf_feature_names,
                       n_top_words, 'Topics in LDA model')

    topic_word_dist = lda_model.components_
    # Clusterize into classes
    # Matrix of documents (rows) with its topic probability (columns):
    all_data_tocluster = lda_model.transform(all_data_t)
    return lda_model, all_data_tocluster, topic_word_dist
