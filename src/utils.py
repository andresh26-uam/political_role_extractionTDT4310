

from pandas.core.frame import DataFrame
from sklearn.decomposition import LatentDirichletAllocation
import seaborn

import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

from src import BEST_ESTIMATORS_ROUTE, DEFAULT_KEYWORD_FILTER_KEEP, \
    DEFAULT_N_KEYWORDS, DEFAULT_N_STDOCS, DEFAULT_N_WORDS_SUMMARIES


def print_best_params(searchmodel: GridSearchCV, name: str) -> None:
    """Prints the best params into the 'bestestimators.txt' file

    Args:
        searchmodel (GridSearchCV): Grid search Model, fitted
        name (str): Name of the classifier to be shown in the txt file
    """
    with open(BEST_ESTIMATORS_ROUTE, "a+") as f:
        print(str(name)+" best params:", file=f)
        print("\t" + str(searchmodel.best_params_), file=f)


def showclusters(data: DataFrame) -> None:
    """Plots the clusters in the 2 first dimensions
    https://seaborn.pydata.org/generated/seaborn.scatterplot.html
    (seaborn snippet code)
    Args:
        data (DataFrame): Dataframe output from the 'clusterize'
            method
    """
    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(8, 8))

    scat = seaborn.scatterplot(
        "component_1",
        "component_0",

        s=50,
        data=data,
        hue="predicted_cluster",
        palette="Set2",
    )

    scat.set_title(
        "Clustering results from "
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    plt.show()


def plot_top_words(model: LatentDirichletAllocation,
                   feature_names: list, n_top_words: int, title: str) -> None:
    """This code is greatly based on the snippet in
    https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
    It plots the topics extracted by the model with the most important words
    describing each one

    Args:
        model (LatentDirichletAllocation): LDA model, pretrained
        feature_names (list): Feature vector
            (words in the TF-IDF vectorizing index)
        n_top_words (int): Number of words to be shown per topic
            (it may be truncated
            to 5 if it is bigger thatn that, in order to see something
            in the graph)
        title (str): Title of the graph
    """
    if len(model.components_) >= 10:
        fig, axes = plt.subplots(len(model.components_)//5,
                                 len(model.components_)//(
                                     len(model.components_)//5),
                                 figsize=(30, 15),
                                 sharex=True)
    else:
        fig, axes = plt.subplots(
            len(model.components_), 1, figsize=(30, 15), sharex=True)

    axes = axes.flatten()

    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-min(n_top_words, 5) - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


def bestestimators_reader():
    with open(BEST_ESTIMATORS_ROUTE) as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            if 'LDA' in lines[i]:
                i += 1
                params_lda = dict(eval(lines[i]))
                for k in params_lda:
                    params_lda[k] = [params_lda[k]]

            if 'KPCA' in lines[i]:
                i += 1
                params_cluster = dict(eval(lines[i]))
                for k in params_cluster:
                    params_cluster[k] = [params_cluster[k]]
            i += 1

    return params_lda, params_cluster


def add_arguments(argparser):
    argparser.add_argument('--use_last_args', '-l', dest='use_last_args',
                           action='store_true',
                           help="""If specified, last args from last
                           execution of train or test will be used instead
                           of whatever other arguments passed""")
    argparser.add_argument('--use_hillary', '-uh', dest='use_hillary',
                           action='store_true',
                           help="""If specified, the Hillary stance detection
                           tweets will be used instead of the Tweepy corpus""")
    argparser.add_argument('--reload_tweet_corpus', '-r',
                           dest='reload_tweet_corpus',
                           action='store_true',
                           help="""If specified, the Tweepy corpus
                            will be regenerated
                            (deleted and refilled with new tweets)""")
    argparser.add_argument('--add_tweets', '-a',
                           dest='add_tweets',
                           action='store_true',
                           help="""If specified, the Tweepy corpus
                            will be enlarged as much as possible
                            (filled with new tweets)""")
    argparser.add_argument('--stdocs', '-std', nargs='?', type=int,
                           default=DEFAULT_N_STDOCS,
                           help="""Declares how many documents from
                            state_union corpus to be selected for the
                            keyword extraction""")
    argparser.add_argument('--random_state', '-rds', nargs='?', type=int,
                           default=42,
                           help="""Random number seed, use 42
                           to replicate the experiments""")
    argparser.add_argument('--n_keywords', '-k', nargs='?', type=int,
                           default=DEFAULT_N_KEYWORDS,
                           help="""Number of keywords to be selected among
                            the state_union corpus. If set to -1,
                            prevously calculated keywords will be used""")
    argparser.add_argument('--sumwords', '-sw', nargs='?', type=int,
                           default=DEFAULT_N_WORDS_SUMMARIES,
                           help="""Specify number of
                           words per summary of a cluster""")
    argparser.add_argument('--keyw_retain', '-kretain', nargs='?', type=float,
                           default=DEFAULT_KEYWORD_FILTER_KEEP,
                           help="""Specify proportion of tweets
                           to keep in the keyword filter""")
    argparser.add_argument('--plot', '-p', action='store_true', dest='plot',
                           help="""Plots intermediate results
                           (clusters, topics)""")
    argparser.add_argument('--keyw_read', '-kread', action='store_true',
                           dest='keyw_read',
                           help="""If specified, try to read previously
                           generated keywords instead of recalculating
                           them from the state_union corpus""")
    argparser.add_argument('--best_params', '-b', action='store_true',
                           dest='best_params',
                           help="""If specified, try to read previously
                           generated best params in {}
                           """.format(BEST_ESTIMATORS_ROUTE))
    return argparser
