

import joblib
import nltk
from pandas.core.frame import DataFrame
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src import DEFAULT_N_WORDS_SUMMARIES, LAST_SUMMARIES_ROUTE,\
    MIN_N_TOP_TWEETS, KEYWORDS_ROUTE, TOP_TWEETS_FILTER
from src.relevance import get_top_tweets
from src.relevance import max_relevance, relevance_score
from nltk.probability import FreqDist


from gensim.summarization import summarize
from gensim.summarization import keywords


def only_nouns(keys):
    ptagkeys = nltk.pos_tag(keys, tagset='universal')
    return [pos_word[0] for pos_word in ptagkeys if pos_word[1] == 'NOUN']


def keyword_extraction(state_union, n_keywords, ndocuments_stateunion):

    if n_keywords < 0:
        # Read the keywords from the file where it were saved before
        try:
            return joblib.load(KEYWORDS_ROUTE)
        except(FileNotFoundError):
            print("ERROR: No valid keywords file found.")
            while(True):
                inp = input("Do you want to select keywords again? [Y/N]: ")
                if inp == 'Y' or inp == 'y':
                    n_keywords = int(input("How many keywords?: "))
                    ndocuments_stateunion = int(input("""How many documents
                     in state_union to see?: """))
                    break
                elif inp == 'N' or inp == 'n':
                    return None
    unions = [state_union.raw(t) for t in np.random.permutation(
        state_union.fileids())[
        :ndocuments_stateunion]]
    # Keyword extraction using gensim 'keywords'
    ks = [only_nouns(keywords(t, words=n_keywords, split=True,
                              lemmatize=True))
          for t in unions if type(t) is str]
    # keywords in a flat list (all in same level)
    flatset_of_keywords = list()
    for lis in ks:
        for word in lis:
            flatset_of_keywords.append(word)
    # We select the most common in all documents
    fd = FreqDist(flatset_of_keywords)
    keywords_to_match = [k[0] for k in fd.most_common(n_keywords)]
    joblib.dump(keywords_to_match, KEYWORDS_ROUTE)
    return keywords_to_match


def summarize_clusters(dataframe_clust: DataFrame, clusters: list,
                       tfidf: TfidfVectorizer, mapping=None,
                       labeled=False,
                       n_words_summaries=DEFAULT_N_WORDS_SUMMARIES) -> None:
    """Summarizes the clusters made

    Args:
        dataframe_clust (DataFrame): Dataframe with predicted classes
        clusters (list): list of predicted cluster for each document in
            the dataframe_clust
        mapping (dict, optional): Mapping of supervised labels (numbers) to
            its meaning (str).
            Those labels are in the 'trueval' column of the
            dataframe_clust.
            This argument is only used for previously
            labeled datasets, as the hillary one.
            Therefore, you don't need to provide the 'trueval' column in
            normal cases.
            The point of this mapping is to see the distribution of those
            labels in each cluster. Defaults to None.
        labeled (bool, optional): This argument is to make the behaviour
            explained before
        to work. Defaults to False.
    """
    with open(LAST_SUMMARIES_ROUTE, "w+") as f:
        pass
    dict_summaries = dict()
    for i in range(0, len(list(set(clusters)))):
        indexes = [j for j, e in enumerate(
            dataframe_clust["predicted_cluster"]) if e == i]
        if (len(indexes) <= 1):
            print("No tweets of cluster ", i, "skipping...")
            continue
        tweets_from_cluster = np.array(
            dataframe_clust["tweets"])[indexes]

        ms = max_relevance(tweets_from_cluster)
        tweet_w_score = [(t, relevance_score(t, ms))
                         for t in tweets_from_cluster]
        n_top = max(
            min(len(tweet_w_score), MIN_N_TOP_TWEETS),
            int(TOP_TWEETS_FILTER*len(tweet_w_score)))
        top_tweets = get_top_tweets(tweet_w_score, n=n_top)
        text = ".\n".join([t['text'] for t in top_tweets])
        print(text)
        summerize = summarize(text, word_count=n_words_summaries)
        print("-----------------------------------------")
        print("Summarizing tweets of class", i, end="")
        dict_summaries[str(i)] = [None, None]
        if labeled:
            truevals = [t['trueval'] for t in tweets_from_cluster]
            mcommon = FreqDist(truevals)
            dict_summaries[str(i)][1] = dict()
            print(", where: ")
            for m in mcommon.keys():
                string = "\t * {:.2f}".format(mcommon.freq(
                    m)*100) + "% of the samples were originally labeled as \""\
                    + str(mapping[m]) + "\", "
                # Save the frequency of the label "mapping[m]" in position
                # 1 of the tuple assigned to the summaries of the cluster i.
                # first value of the tuple are the actual summaries:
                dict_summaries[str(i)][1][mapping[m]] = mcommon.freq(m)*100
                print(string)
        else:
            print()
        print(summerize)
        dict_summaries[str(i)][0] = summerize
        print("-----------------------------------------")
        print()
    with open(LAST_SUMMARIES_ROUTE, "a+") as f:
        print(dict_summaries, file=f)
