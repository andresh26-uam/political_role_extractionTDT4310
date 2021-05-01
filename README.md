# Extracting Political Roles from Tweets
Code for the subject TDT4310 course project, named "Extracting political roles from tweets", by Andrés Holgado Sánchez, 2021
--------
## Aim of the project
This project tries to grasp political content on a bunch of tweets, which neednd't be already selected to be "political" nor even told how political tweets are. Political highlighting is made by using keyword extraction from historical political statements and then, using Latent Dirichlet Allocation, perform a Topic modelling on the tweets to match those topics to our selected keywords in an original manner. After that, a clustering process is done on the remaining tweets, resulting into different political stances if the process was correctly made. To check the results, a summarizing process is done to return a summary of each cluster of tweets. How to de that, is explained in [Installation and running instructions](instrun)
<a href="instrun"></a>
## Repository contents

In this repository you can find all the source code and sources for the project, except the paper, which is not made public yet.
Folder `src/` contains the source code and in top level you can find 2 scripts: `test.py`and `train.py`. Another file, which should not be directly changed is called "bestestimators.txt", a file containing info on the best found hyperparameters for every estimator used in the model.
