import joblib
from src.cleaning import clean_data_set, preprocess_tweet
from src import HILLARY_LABEL_MAPPING, LAST_ARGS_ROUTE, \
    MAX_TWEEPY_TWEETS, TRAINED_CLUSTERER_ROUTE, \
    TRAINED_TFIDF_ROUTE, TRAINED_LDA_ROUTE, KEYWORDS_ROUTE
from src.clustering import clusterize
from src.corpus_manager import TweetCorpus

from src.pretrained import keyword_extraction, summarize_clusters
from src.topicmodel import topic_modelling
from src.utils import add_arguments

from train import keyword_filter
from nltk.corpus import state_union


def run_test(all_test, keywords_to_match, tfidf, mapping,
             labeled, random_st=42, do_plot=True):
    print("Cleaning tweets...")
    all_test_cl = clean_data_set(all_test)
    print("Done")
    print("Vectorizing tweets...")
    all_test_pp = [preprocess_tweet(t['text']) for t in all_test_cl]

    all_test_t = tfidf.transform(all_test_pp)
    print("Done")
    print("Topic Modelling...")
    model, topiqued_test, topics = topic_modelling(
        all_test_t,
        tfidf, len(keywords_to_match),
        retrain=False, plot=do_plot)
    print("Done")
    print("Keyword filtering...")
    topiqued_test, all_test_cl, all_test_t = keyword_filter(
        topiqued_test, all_test_cl, all_test_t,
        keywords_to_match, topics, n_keywords, tfidf, retain=retain_arg)
    print("Done")
    print("Clusterinzing...")
    clusterer, clusterized_dataframe, clusters = clusterize(
        all_test_cl, all_test_t, retrain=False,
        random_st=random_st, plot=do_plot)
    print("Done")

    summarize_clusters(clusterized_dataframe, clusters,
                       tfidf, mapping, labeled)


if __name__ == "__main__":

    from argparse import ArgumentParser
    argparser = ArgumentParser(
        description="""Test the model using the data specified
         (by default the Tweepy corpus, which generalizes
         better than Hillary's). Models are supposed to be
         already trained. They can be found at:
         - {} (keywords)\n
         - {} (TF-IDF vectorizer)\n
         - {} (LDA model)\n
         - {} (Clusterizer: Kmeans + KPCA)\n""".format(
            KEYWORDS_ROUTE,
            TRAINED_TFIDF_ROUTE,
            TRAINED_LDA_ROUTE,
            TRAINED_CLUSTERER_ROUTE
        ))
    argparser = add_arguments(argparser)

    args = argparser.parse_args()
    if args.use_last_args:
        args = joblib.load(LAST_ARGS_ROUTE)
    random_st = args.random_state
    big_corpus = args.use_hillary or False
    ndocuments_stateunion = args.stdocs
    n_keywords = args.n_keywords
    n_words_summaries = args.sumwords
    retain_arg = args.keyw_retain or False
    plot_arg = args.plot or False
    read_keyword_arg = args.keyw_read or False
    joblib.dump(args, LAST_ARGS_ROUTE)

    keywords_to_match = keyword_extraction(
        state_union, -1, ndocuments_stateunion)
    if keywords_to_match is None:
        print("Aborting: No keywords selected")
    print("Keywords selected: \n", keywords_to_match)
    # The corpus here might go for the TweetEval corpus or the Tweepy one,
    # depending on the argument --use_hillary (1 goes for hillary, 0 otherwise)
    corpus = TweetCorpus(reload=args.reload_tweet_corpus or False,
                         add_more=args.add_tweets,
                         n_tweets=MAX_TWEEPY_TWEETS,
                         keywords_filter=keywords_to_match)

    # We read it. big_corpus = (--use_hillary == 1)
    if big_corpus:
        print("Reading Hillary Corpus... ")
        all_data, all_test = corpus.read_big_corpus()
    else:
        print("Reading Tweepy Corpus... ")
        all_data, all_test = corpus.read_corpus()
    print("Done")
    tfidf = joblib.load(TRAINED_TFIDF_ROUTE)
    run_test(all_test, keywords_to_match, tfidf, mapping=HILLARY_LABEL_MAPPING,
             labeled=big_corpus, random_st=random_st, do_plot=plot_arg)
