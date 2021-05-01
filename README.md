# Extracting Political Roles from Tweets
Code for the subject TDT4310 course project, named "Extracting political roles from tweets", by Andrés Holgado Sánchez, 2021
--------
## Aim of the project
This project tries to grasp political content on a bunch of tweets, which neednd't be already selected to be "political" nor even told how political tweets are. Political highlighting is made by using keyword extraction from historical political statements and then, using Latent Dirichlet Allocation, perform a Topic modelling on the tweets to match those topics to our selected keywords in an original manner. After that, a clustering process is done on the remaining tweets, resulting into different political stances if the process was correctly made. To check the results, a summarizing process is done to return a summary of each cluster of tweets. How to de that, is explained in [Installation and running instructions](installation-and-running-instructions)
## Repository contents

In this repository you can find all the source code and sources for the project, except the paper, which is not made public yet.
Folder `src/` contains the source code and in top level you can find 2 scripts: `test.py`and `train.py`. Another file, which should not be directly changed is called "bestestimators.txt", a file containing info on the best found hyperparameters for every estimator used in the model.
## Installation and running instructions
The project uses 3 datasets, 2 of them are a choice of which tweets to use as source for the experiments. One is called "Hillary's dataset" and the other is "Tweepy dataset". The first one comes from the stance detection task from [TweetEval 2020](https://github.com/cardiffnlp/tweeteval) paper. And the other is a corpus of tweets extracted using the [Tweepy](https://www.tweepy.org/) library. You can reload that corpus of tweets, but not enlarge it, unless doing so manually putting tweets in the 'corpus/' folder respecting the conventions of the other tweets.
The other dataset is fixed. It is the "state_union" corpus from [NLTK](https://www.nltk.org/), used as a reference for political statement keywords. You can specify how many keywords to use, how many documents to gather from that corpus, etc.
After that necessary introduction, these are the general instructions for running the project:
- Download/clone the repository
- Run `test.py` or `train.py`depending on what you are going to do. This is the helping `python3 test.py -h` output as a tutorial on how to use the command:
```console
foo@bar:~$ python3 train.py -h
usage: train.py [-h] [--use_hillary USE_HILLARY]
                [--reload_tweet_corpus RELOAD_TWEET_CORPUS]
                [--stdocs [STDOCS]] [--random_state [RANDOM_STATE]]
                [--n_keywords [N_KEYWORDS]] [--sumwords [SUMWORDS]]
                [--keyw_retain [KEYW_RETAIN]]

Train the model using the data specified (by default the Tweepy corpus, which
generalizes better than Hillary's)

optional arguments:
  -h, --help            show this help message and exit
  --use_hillary USE_HILLARY
                        If specified, the Hillary stance detection tweets will
                        be used instead of the Tweepy corpus
  --reload_tweet_corpus RELOAD_TWEET_CORPUS
                        If specified, the Tweepy corpus will be regenerated
                        (deleted and refilled with new tweets)
  --stdocs [STDOCS]     Declares how many documents from state_union corpus to
                        be selected for the keyword extraction
  --random_state [RANDOM_STATE]
                        Random number seed, use 42 to replicate the
                        experiments
  --n_keywords [N_KEYWORDS]
                        Number of keywords to be selected among the
                        state_union corpus
  --sumwords [SUMWORDS]
                        Specify number of words per summary of a cluster
  --keyw_retain [KEYW_RETAIN]
                        Specify proportion of tweets to keep in the keyword
                        filter
```
- Instructions for `test.py`:
