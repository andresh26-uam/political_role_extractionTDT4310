from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import demoji
import nltk
import re
from nltk.corpus import stopwords
THRESHOLD = 5  # Minimum number of characters to consider a tweet to be
# able to be correctly classified.


def clean_data_set(list_of_tweets):
    """Cleans the given dataset of tweets

    Args:
        list_of_tweets (list): List of raw tweets having the
        attribute 'text' in it

    Returns:
        list: Same list, with cleaned contents
    """
    returning = list()
    for t in list_of_tweets:
        twe = clean_or_discard_t(t)
        if twe['text'] is None:
            continue
        returning.append(twe)

    return returning


def cleanText(text):
    """Cleans raw text, getting rid of links and
    other silly punctuation words

    Args:
        text (str): String of text (a tweet)

    Returns:
        str: Cleaned tweet
    """

    emos = demoji.findall(text)
    for e, te in emos.items():
        text = re.sub(re.compile(e), te, text)

    text = re.sub(r'http\S+', r'', text)
    text = re.sub(r'RT[\:]*', r'', text)
    text = re.sub(r'@user', r'', text)
    text = re.sub(r'@[A-Za-z0-9]+', r'', text)
    
    text = re.sub(r'’|‘', '\'', text)
    text = re.sub(r'“|”', '\"', text)
    # TweetEval dataset tweets always have this termination,
    #  indicating they come from SemEval-2016 dataset(?),
    # Anyway, need to get rid of it
    text = re.sub(r'#SemST', r'', text)
    text = re.sub(r'#[A-Za-z0-9]+', r'', text)
    text = text.lower()
    return text


def clean_or_discard(tweet):
    """Cleans a tweet text, but also returns None if
    its resulting length is too small to be able to be
    classified correctly

    Args:
        tweet (str):tweet

    Returns:
        str: Same str, cleaned, or None
    """
    tweet = cleanText(text=tweet)
    if len(tweet) < THRESHOLD:
        return None

    return tweet


def clean_or_discard_t(tweet_complete):
    """Same as before, but applied to a complete tweet,
    which is a dict with the attribute 'text' in it. It changes
    that attribute

    Args:
        tweet (dict): Dictionary of tweet

    Returns:
        dict: Same dict, changing the 'text' value
    """
    tweet_complete['text'] = clean_or_discard(tweet_complete['text'])
    return tweet_complete


lemmatizer = WordNetLemmatizer()


stopwords_list = stopwords.words('english')


def preprocess_tweet(tweet_cleaned):
    tokenizer = nltk.tokenize.TweetTokenizer()

    tc = tokenizer.tokenize(tweet_cleaned)
    t = [word.lower() for word in tc
         if word not in stopwords_list]
    tagged_t = nltk.tag.pos_tag(t)
    # I need a conversion of nltk pos tags to wordnet's:

    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    # This converts the pairs (word, nltk_pos_tag) to the pairs:
    # (word, wordnet_pos_tag)
    tagged_t = map(lambda tag: (tag[0], tag_dict.get(
        tag[1][0].upper(), wordnet.NOUN)), tagged_t)

    t = " ".join([lemmatizer.lemmatize(tuple[0], tuple[1])
                  for tuple in tagged_t])
    return t
