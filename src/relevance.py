import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from textblob.blob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer


def relevance_score(tweet: str, max_score: float) -> float:
    """Computes the 'relevance' of a tweet.
    Also uses the TextBlob subjectitvity measure to make subjective tweets
    more relevant. We don't want to highlight news articles, for example

    Args:
        tweets (str): Tweet (cleaned)

    Returns:
        float: Relevance (social importance of the tweets list)
    """
    return ((tweet['likes']+tweet['rts'])*tweet['foll']/max_score)*(
        (TextBlob(tweet['text']).subjectivity)**5)


def representatives(tweets: list, n: int, tfidf: TfidfVectorizer) -> list:
    """Returns the representative tweets from a list of them,
    They are selected according to their cosine_similarity
    to every other tweet.
    The ones with the most cosine_similarity to every other tweet are selected
    (thus, they are the ones that held the most similarity to the whole group,
    the most representatives)

    Args:
        tweets (list): list of tweets (cleaned)
        n (int): number of representatives to keep in the returned list
        tfidf (TfidfVectorizer): TF-IDF vectorizer
    Returns:
        list: representatives
    """
    tfidf_tweets = tfidf.transform([t['text'] for t in tweets])
    tcosine = [(tweets[i], sum([cosine_similarity(tfidf_tweets[i], t2)
                                for t2 in tfidf_tweets]))
               for i in range(0, len(tweets))]

    return get_top_tweets(tcosine, n=n)


def get_top_tweets(tweets_w_score: list, n: int) -> list:
    """Get top n players by score

    Args:
        tweets_w_score (list): List of tuples (tweet(str), score(float))
        n (int): Number of tweets to keep as "top" tweets according to
                 their score
        (descending order)

    Returns:
        list: list of n top scored tweets
    """
    indexes = [(i, e[1]) for i, e in enumerate(tweets_w_score)]
    top = sorted(indexes, key=lambda x: x[1], reverse=True)[:n]

    return list(np.array(
                    [t[0] for t in tweets_w_score]
                )[np.array([ind[0] for ind in top])])


def max_relevance(tweets: list) -> float:
    """Computes the maximum 'relevance' of tweets

    Args:
        tweets (list): Tweets (cleaned)

    Returns:
        float: Max possible relevance (social importance of the tweets list)
    """
    df = pd.DataFrame(list(tweets), columns=['text', 'rts', 'likes', 'foll'])
    return (max(df['likes'])+max(df['rts']))*max(df['foll'])
