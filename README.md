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
- Check the configurations available in the file [src/__init__.py](https://github.com/andresh26-uam/political_role_extractionTDT4310/src/__init__.py)
- Run `test.py` or `train.py`depending on what you are going to do with the model. This is the helping `python3 train.py -h` output as a tutorial on how to use the command:
```console
foo@bar:~$ python3 train.py -h
usage: train.py [-h] [--use_last_args] [--use_hillary] [--reload_tweet_corpus]
                [--test_ratio [TEST_RATIO]] [--add_tweets] [--stdocs [STDOCS]]
                [--random_state [RANDOM_STATE]] [--n_keywords [N_KEYWORDS]]
                [--sumwords [SUMWORDS]] [--keyw_retain [KEYW_RETAIN]] [--plot]
                [--keyw_read] [--best_params]

Train the model using the data specified(by default the Tweepy corpus, which generalizesbetter than Hillary's).
        Trained models are overriden inthese locations:
         - /home/andres/NTNU/NLP/PROJECT/political_role_extractionTDT4310/src/../pkl/keywords.pkl (keywords)
         - /home/andres/NTNU/NLP/PROJECT/political_role_extractionTDT4310/src/../pkl/trained_tfidf.pkl (TF-IDF vectorizer)
         - /home/andres/NTNU/NLP/PROJECT/political_role_extractionTDT4310/src/../pkl/trained_lda.pkl (LDA model)
         - /home/andres/NTNU/NLP/PROJECT/political_role_extractionTDT4310/src/../pkl/trained_clusterer.pkl (Clusterizer: Kmeans + KPCA)
         

optional arguments:
  -h, --help            show this help message and exit
  --use_last_args, -l   If specified, last args from last execution of train or test will be used instead of whatever other arguments passed
  --use_hillary, -uh    If specified, the Hillary stance detection tweets will be used instead of the Tweepy corpus
  --reload_tweet_corpus, -r
                        If specified, the Tweepy corpus will be regenerated(deleted and refilled with new tweets)
  --test_ratio [TEST_RATIO], -t [TEST_RATIO]
                        Test samples (ratio vs total corpus)
  --add_tweets, -a      If specified, the Tweepy corpus will be enlarged as much as possible(filled with new tweets)
  --stdocs [STDOCS], -std [STDOCS]
                        Declares how many documents from state_union corpus to be selected for the keyword extraction
  --random_state [RANDOM_STATE], -rds [RANDOM_STATE]
                        Random number seed, use 42 to replicate the experiments
  --n_keywords [N_KEYWORDS], -k [N_KEYWORDS]
                        Number of keywords to be selected among the state_union corpus. If set to -1, prevously calculated keywords will be used
  --sumwords [SUMWORDS], -sw [SUMWORDS]
                        Specify number of words per summary of a cluster
  --keyw_retain [KEYW_RETAIN], -kretain [KEYW_RETAIN]
                        Specify proportion of tweets to keep in the keyword filter
  --plot, -p            Plots intermediate results (clusters, topics)
  --keyw_read, -kread   If specified, try to read previously generated keywords instead of recalculating them from the state_union corpus
  --best_params, -b     If specified, try to read previously generated best params in /home/andres/NTNU/NLP/PROJECT/political_role_extractionTDT4310/src/../bestestimators.txt
```
- To run `test.py`, it is mandatory having previously executed `train.py`, as this last script makes some files and models trained to testing. After one execution of `train.py`, several `test.py` can be run over the same trained models. 
- Testing is very fast, normally, but training might take a lot of time depending mainly on the parameter grids supplied in the files [src/topicmodel.py](https://github.com/andresh26-uam/political_role_extractionTDT4310/src/topicmodel.py) and [src/clustering.py](https://github.com/andresh26-uam/political_role_extractionTDT4310/src/clustering.py)
