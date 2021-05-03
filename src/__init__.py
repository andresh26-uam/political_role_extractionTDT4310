
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
DEFAULT_N_WORDS_SUMMARIES = 140

TRAINED_CLUSTERER_ROUTE = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../trained_clusterer.pkl")
TRAINED_LDA_ROUTE = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../trained_lda.pkl")
TRAINED_TFIDF_ROUTE = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../trained_tfidf.pkl")
KEYWORDS_ROUTE = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../keywords.pkl")
LAST_ARGS_ROUTE = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../args.pkl")
BEST_ESTIMATORS_ROUTE = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "../bestestimators.txt")
HILLARY_LABEL_MAPPING = {'0': 'neutral', '1': 'against', '2': 'favor'}
