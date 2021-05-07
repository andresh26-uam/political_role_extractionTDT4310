
import os


N_FEATURES = 10000
# Max number of iterations in LDA algorithm
LDA_MAXITER = 6
# Maximum number of tweets to gather in tweepy corpus.
# It will throw an error 429 if too many tweets are specified
MAX_TWEEPY_TWEETS = 1200
# Default number of keywords,
# (can be specified with --n_keywords arg in test/train.py)
DEFAULT_N_KEYWORDS = 20
# Declares how many documents
# from state_union corpus to be selected for the keyword extraction
DEFAULT_N_STDOCS = 100
# Specify proportion of tweets to keep in the keyword filter
DEFAULT_KEYWORD_FILTER_KEEP = 0.33
# Maximum number of words of a summarization output
DEFAULT_N_WORDS_SUMMARIES = 140
DEFAULT_TEST_RATIO = 0.3
# Proportion of tweets to consider from each cluster to be in the summarization
# input
TOP_TWEETS_FILTER = 0.25
# Minimum number of tweets to consider from each cluster
# to be in the summarization input, if that cluster entails more than
# this number of tweets but after applying the TOP_TWEETS_FILTER,
# the remaining are smaller number than this value
MIN_N_TOP_TWEETS = 10
TRAINED_CLUSTERER_ROUTE = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../pkl/trained_clusterer.pkl")
TRAINED_LDA_ROUTE = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../pkl/trained_lda.pkl")
TRAINED_TFIDF_ROUTE = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../pkl/trained_tfidf.pkl")
KEYWORDS_ROUTE = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../pkl/keywords.pkl")
LAST_ARGS_ROUTE = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../pkl/args.pkl")
BEST_ESTIMATORS_ROUTE = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../bestestimators.txt")
HILLARY_LABEL_MAPPING = {'0': 'neutral', '1': 'against', '2': 'favor'}
LAST_SCORES_ROUTE = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../scores.txt")
EXPERIMENT_RESULTS_ROUTE = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../pkl/experiment_scores.pkl")
EXPERIMENT_RESULTS_ROUTE_TEST = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../pkl/experiment_scores_test.pkl")
EXPERIMENT_RESULTS_ROUTE_H_TEST = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../pkl/experiment_scores_h_test.pkl")

LAST_SUMMARIES_ROUTE = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../last_summaries.txt")
COHERENCE = 0
PERPLEXITY = 1
INV_SCORE = 2
SILHOUETTE = 3
SCORE_NAMES = ['LDA coherence', 'LDA perplexity',
               'KPCA inverse score', 'Kmeans silhouette']
