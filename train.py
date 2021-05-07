# %%
from argparse import RawTextHelpFormatter
from src.utils import add_arguments, bestestimators_reader
from src.topicmodel import TRAIN_PARAM_GRID_LDA, topic_modelling
from typing import Any, Iterable, List
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


import demoji
import os
from sklearn.metrics import silhouette_score
import numpy as np


from src.cleaning import preprocess_tweet, clean_data_set
from src.clustering import TRAIN_PARAM_GRID_CLUSTERER, clusterize
from src.corpus_manager import TweetCorpus

from src.pretrained import keyword_extraction, summarize_clusters
from nltk.corpus import state_union
from src import BEST_ESTIMATORS_ROUTE, DEFAULT_KEYWORD_FILTER_KEEP,\
    HILLARY_LABEL_MAPPING,\
    LAST_ARGS_ROUTE, LAST_SCORES_ROUTE, \
    MAX_TWEEPY_TWEETS, N_FEATURES, TRAINED_CLUSTERER_ROUTE, \
    TRAINED_TFIDF_ROUTE, TRAINED_LDA_ROUTE, KEYWORDS_ROUTE


# %%


def calculate_score(imp: list, dats: list) -> float:
    """Calculates the score of a tweet
    given the importance of each topic
    and the topic distribution of that tweet

    Args:
        imp (list): importance of topic
        dats (list): topic dictribution in a tweet

    Returns:
        float: score of the tweet with such topic distribution
    """
    score = 0
    for i in range(0, len(imp)):
        score += imp[i]*dats[i]
    return score


def keyword_filter(data: List[List[float]], tweets: List[dict],
                   tweets_vectorized: List[List[Any]], keys: Iterable,
                   topics: list, n_keywords: int,
                   tfidf: TfidfVectorizer, retain=DEFAULT_KEYWORD_FILTER_KEEP):
    """This function selects the 'retain' proportion of tweets
    having the biggest 'keyword score' which is a proposed
    measure of the significance of a tweet given a set of keywords
    and topics extracted via LDA.

    Args:
        data (List[List[float]]): list where each entry is
            the topic distribution
            of a tweet supplied in the 'tweets' argument
        tweets (List[dict]): list of tweets (complete,
            with metadata but cleaned)
        tweets_vectorized (List[List[Any]]): Vectorized tweets
        keys (Iterable): keywords
        n_keywords (int): number of keywords
        topics (list): Topics (LDAmodel.components_)
        tfidf (TfidfVectorizer): TF-IDF vectorizer
        retain (float, optional): Proportion of tweets to keep.
        Defaults to DEFAULT_KEYWORD_FILTER_KEEP (found in __init__.py).

    Returns:
        tuple(list, list, list): Tuple of:
        - Selected tweets ready to clusterize,
        - Cleaned complete tweets selected,
        - Tokenized tweets selected
    """
    tf_feature_names = tfidf.get_feature_names()
    importance_of_topics = list()
    tweet_score = list()
    for t in topics:
        top_features_ind = t.argsort()[:-n_keywords - 1:-1]
        top_features = [tf_feature_names[i] for i in top_features_ind]
        importance_of_topics.append(
            len([m for m in top_features if m in keys])/len(topics))
    for i in range(0, len(tweets)):
        tweet_score.append(
            (i, calculate_score(importance_of_topics, data[i])))

    tweet_score.sort(key=lambda x: x[1], reverse=True)

    # Most political related tweet indices are stored now in tweet_score
    indices = [t[0] for t in tweet_score[0:int(retain*len(data))]]
    # We retain the indices of the int(retain*len(data)) most scored tweets
    topic_twdist = np.array(data)[indices]
    filtered_tweets = np.array(tweets)[indices]
    filtered_vec_tweets = tweets_vectorized[indices]
    return topic_twdist, filtered_tweets, filtered_vec_tweets


# %%


if __name__ == "__main__":

    from argparse import ArgumentParser

    argparser = ArgumentParser(
        description="""Train the model using the data specified\
(by default the Tweepy corpus, which generalizes\
better than Hillary's).\n\tTrained models are overriden in\
these locations:
         - {} (keywords)
         - {} (TF-IDF vectorizer)
         - {} (LDA model)
         - {} (Clusterizer: Kmeans + KPCA)
         """.format(
            KEYWORDS_ROUTE,
            TRAINED_TFIDF_ROUTE,
            TRAINED_LDA_ROUTE,
            TRAINED_CLUSTERER_ROUTE), formatter_class=RawTextHelpFormatter)
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

    param_clust = TRAIN_PARAM_GRID_CLUSTERER
    param_lda = TRAIN_PARAM_GRID_LDA
    if args.best_params:
        param_lda, param_clust = bestestimators_reader()
    joblib.dump(args, LAST_ARGS_ROUTE)

    tfidf = TfidfVectorizer(max_df=0.95, min_df=2,
                            max_features=N_FEATURES,
                            stop_words='english')
    demoji.download_codes()  # DO THIS only FIRST TIME EXECUTING

    
    # Now we have to extract the keywords from political statements. Number of
    # keywords is a parameter useful to optimize and it is shown in graphs
    # the different coherence and silhouette scores
    keywords_to_match = keyword_extraction(
        state_union, n_keywords, ndocuments_stateunion)
    # We may find less keywords than the proposed by the user:
    n_keywords = len(keywords_to_match)
    print("Keywords selected: \n", keywords_to_match)
    # The corpus here might go for the TweetEval corpus or the Tweepy one,
    # depending on the argument --use_hillary (1 goes for hillary, 0 otherwise)
    corpus = TweetCorpus(reload=args.reload_tweet_corpus,
                         add_more=args.add_tweets,
                         n_tweets=MAX_TWEEPY_TWEETS,
                         keywords_filter=keywords_to_match)

    # We read it. big_corpus = (--use_hillary == 1)
    if big_corpus:
        print("Reading Hillary Corpus... ")
        all_data, all_test = corpus.read_big_corpus()
        all_data = all_data+all_test # with Hillary we don't bother here

    else:
        print("Reading Tweepy Corpus... ")
        all_data, all_test = corpus.read_corpus(ratio_train=1-args.test_ratio)
    print("Done.")
    # This file saves the hyperparameter selection.
    # After training, it is overriden with better values
    # Use it with the test.py file command
    if os.path.exists(BEST_ESTIMATORS_ROUTE):
        os.remove(BEST_ESTIMATORS_ROUTE)
    if os.path.exists(LAST_SCORES_ROUTE):
        os.remove(LAST_SCORES_ROUTE)
    print("Best params file and last scores were deleted. Training started...")
    # and clean the returned tweets
    print("Cleaning tweets...")
    all_data = clean_data_set(all_data)
    # Preprocessing
    print("Done.")
    print("Vectorizing tweets...")
    all_data_pp = [preprocess_tweet(t['text']) for t in all_data]
    # TF-IDF transform
    all_data_t = tfidf.fit_transform(all_data_pp)
    print("Done.")
    print("Saving Vectorizer for testing...")
    joblib.dump(tfidf, TRAINED_TFIDF_ROUTE)
    print("Done.")
    
    # I will use now LDA (Latent Dirichlet Allocation) to identify
    # topics of the tweets.
    # Then, clusterize the documents filtered by the frequency of use of
    # those keywords.
    # See the keyword_filter function that does the job.
    # https://www.youtube.com/watch?v=BuMu-bdoVrU
    # http://ai.stanford.edu/~ang/papers/jair03-lda.pdf
    print("Topic Modelling...")
    lda_model, all_data_tocluster, topics = topic_modelling(
        all_data_t, tfidf, n_keywords, param_grid=param_lda,
        retrain=True, plot=plot_arg,
        random_seed=random_st)
    print("Done.")
    # Get the most interesting topiqued documents according to our political
    # keywords as a whole:
    print("Keyword filtering...")
    all_data_tocluster, all_data, all_data_t = keyword_filter(
        all_data_tocluster, all_data, all_data_t,
        keywords_to_match, topics, n_keywords, tfidf, retain=retain_arg)
    print("Done.")
    print("Clusterinzing...")
    clusterer, data_preprocessed_nooutlier, clusters = clusterize(
        all_data, all_data_t, random_st=random_st,
        param_grid=param_clust, retrain=True, plot=plot_arg)
    print("Done.")

    print("Final Silhouette Score\n: ",
          silhouette_score(
              data_preprocessed_nooutlier[['component_{}'.format(i)
                                           for i in range(
                  clusterer['preprocessor'].get_params()['n_components'])
              ]],
              clusterer["kmeans"].labels_, metric='cosine')
          )
    summarize_clusters(data_preprocessed_nooutlier,
                       clusters, tfidf, mapping=HILLARY_LABEL_MAPPING,
                       labeled=big_corpus,
                       n_words_summaries=n_words_summaries)
