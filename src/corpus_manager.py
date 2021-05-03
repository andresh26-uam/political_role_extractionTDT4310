import shutil
import os
import json
from numpy import random
from tweepy import OAuthHandler, API, Cursor
consumer_key = "FpVCvvbgR1SPYt66b69GoTope"
consumer_secret = "j3h6nvHvqOahtba3qZTvGyV2E3P39AkLWufblOw00mfzXql47m"
access_token = "1352916185726246912-dfkO6DN8syQs2peUxAKjNGyg67mfs3"
access_token_secret = "RKBYpnvETZo5Knun7qP9vRjkUZkb1jKILHIPmHcciaxGf"
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = API(auth)

# All tweets with less information will be discarded as it's a nonsense
# to fit the model there (will cause randomization the logic of the model)

PRE_CLEAN_THRESHOLD = 20


def add_tweets_to_corpus(iter, user_dict, qid):

    for tweet in iter:
        if str(tweet._json['user']['id']) in user_dict:
            continue
        else:
            user_dict.add(str(tweet._json['user']['id']))
        if str(tweet.id) in user_dict:
            continue
        else:
            user_dict.add(str(tweet.id))

        # If the tweet is a RT, might end with /u2026 (...),
        # This snippet solves the issue
        # (https://stackoverflow.com/questions/52431763/how-to-get-full-text-of-tweets-using-tweepy-in-python)
        status = tweet
        if 'extended_tweet' in status._json:
            status_json = status._json['extended_tweet']['full_text']
        elif 'retweeted_status' in status._json:
            if 'extended_tweet' in status._json['retweeted_status']:
                tweet_real = status._json['retweeted_status']['extended_tweet']
                status_json = tweet_real['full_text']
            else:
                status_json = status._json['retweeted_status']['full_text']

        else:
            status_json = status._json['full_text']

        if len(status_json) < PRE_CLEAN_THRESHOLD:
            continue

        token_results = {
            "text": status_json,
            "user": tweet._json['user']['id'],
            "likes": tweet.favorite_count,
            "rts": tweet.retweet_count,
            "foll": tweet._json['user']['followers_count'],
            'id': tweet.id
        }

        with open(os.path.join(
                TweetCorpus.CORPUS_ROUTE,
                "Tweet_"+str(qid)+".txt"), 'w') as f:

            json.dump(token_results, f)
        qid += 1
    return qid, user_dict


class TweetCorpus():
    # This is the route for the new corpus. Should be
    # the one "<this_dir>/corpus/"
    CORPUS_ROUTE = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), '../corpus/')
    # This other route is where to find the Hillary dataset
    # the one "<this_dir>/corpus/"
    BIG_CORPUS_ROUTE = os.path.join(os.path.dirname(
        os.path.abspath(__file__)),
        '../tweeteval-main/datasets/stance/hillary')

    def create_corpus(self):
        """Generates a Tweepy corpus, saving it in CORPUS_ROUTE
        """

        qid = len(
            [name for name in os.listdir(TweetCorpus.CORPUS_ROUTE)])
        if qid > 0:
            # In order not to repeat tweets...
            rdata, rtest = self.read_corpus()
            rdata = rdata+rtest
            print(rdata[0])
            user_dict = set([str(tweet['id']) for tweet in rdata]).union(
                set([str(tweet['user']) for tweet in rdata]))
        else:
            user_dict = set()
        for k in self.keywords:
            query = str(k) + \
                """%20-is%3Anull_cast%20-is%3Aquote%20-is%3Aretweet%20-is%3Averified%20-is%3Areply%20-has%3Ahashtags%20-has%3Amentions%20-has%3Amedia"""

            iter = Cursor(api.search, q=query, lang="en",
                          result_type="mixed", tweet_mode="extended",
                          include_retweets=False
                          ).items(self.n_tweets//len(self.keywords))

            qid, user_dict = add_tweets_to_corpus(iter, user_dict, qid)

        user_dict.clear()
        return

    def __init__(self, reload: bool, add_more=True, n_tweets=500,
                 keywords_filter=None):
        """This constructor makes a new corpus

        Args:
            reload (bool): Put this to true to override the previous corpus
                with new data
            n_tweets (int, optional): Specifies how many tweets
            (maximum) to keep in the corpus Defaults to 500. It can go below
            that number due to shorter tweets than the
            threshold specified and various filter mismatches.
            keywords_filter (list, optional): Defaults to None. Indicates the
            keywords that are going to be matched in the queries
            to find political tweets
        """
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = API(auth)
        self.n_tweets = n_tweets
        self.keywords = keywords_filter
        if reload:
            shutil.rmtree(TweetCorpus.CORPUS_ROUTE, ignore_errors=True)
        if reload or add_more:
            os.makedirs(TweetCorpus.CORPUS_ROUTE, exist_ok=True)
            self.create_corpus()

    def read_corpus(self, ratio_train=0.7) -> tuple:
        """Reads the corpus of Tweepy tweets into a list of tweets which are
        dict's like this:
        - 'user': (user id)
        - 'id': (tweet id)
        - 'likes': number of favorites received
        - 'rts': number of retweets
        - 'foll': number of followers
        - 'text': actual text of the tweet
        Args:
            ratio_train (float, optional): Ratio of train versus test size.
            Defaults to 0.7.

        Returns:
            tuple: train,test datasets
        """
        n_docs = len(
            [name for name in os.listdir(TweetCorpus.CORPUS_ROUTE)])
        tweets = [0]*n_docs

        for i in range(n_docs):
            path = os.path.join(TweetCorpus.CORPUS_ROUTE,
                                'Tweet_'+str(i)+".txt")

            with open(path, 'r') as f:
                tweet = json.load(f)
                tweets[i] = tweet
        train_data = tweets[0:int(len(tweets)*ratio_train)]
        test_data = tweets[int(len(tweets)*ratio_train):len(tweets)]
        return train_data, test_data

    def read_big_corpus(self) -> tuple:
        """Reads the Hillary corpus, in the same format as before

        Returns:
            tuple: Train, test sets
        """
        train_l = list()
        path = os.path.join(TweetCorpus.BIG_CORPUS_ROUTE,
                            'train_labels'+".txt")
        train_l = open(path).read().split("\n")[:-1]
        test_l = list()
        path = os.path.join(TweetCorpus.BIG_CORPUS_ROUTE,
                            'test_labels'+".txt")
        test_l = open(path).read().split("\n")[:-1]

        train_t = list()
        path = os.path.join(TweetCorpus.BIG_CORPUS_ROUTE,
                            'train_text'+".txt")
        train_t = open(path).read().split("\n")[:-1]
        test_t = list()
        path = os.path.join(TweetCorpus.BIG_CORPUS_ROUTE,
                            'test_text'+".txt")
        test_t = open(path).read().split("\n")[:-1]

        train = [{'text': train_t[i], 'trueval': train_l[i],
                  'likes': 1000, 'rts': 1000, 'foll': 1000,
                  'user': str(random.randint(low=0, high=1000000)),
                  'id': str(random.randint(low=0, high=100000000))}
                 for i in range(0, len(train_l))]
        test = [{'text': test_t[i], 'trueval': test_l[i],
                 'likes': 1000, 'rts': 1000, 'foll': 1000,
                 'user': str(random.randint(low=0, high=1000000)),
                 'id': str(random.randint(low=0, high=100000000))}
                for i in range(0, len(test_l))]
        return train, test
