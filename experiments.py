import os
from src import EXPERIMENT_RESULTS_ROUTE, \
    EXPERIMENT_RESULTS_ROUTE_H_TEST,\
    EXPERIMENT_RESULTS_ROUTE_TEST,\
    LAST_SCORES_ROUTE, LAST_SUMMARIES_ROUTE, SCORE_NAMES

import numpy as np
import matplotlib.pyplot as plt
import subprocess
import joblib


def read_last_scores(prev_scores):
    """Reads the last scores after training or testing
    And also reads the last summaries made. Result
    is added to the previous dictionary of scores of
    a running set of experiments

    Args:
        prev_scores (dict): Previous experiment scores

    Returns:
        dict: Report of prev_scores and the newly read ones
         in a new dictionary
    """
    with open(LAST_SCORES_ROUTE, "r") as f:

        lines = f.readlines()
        i = 0
        while i < len(lines):
            li = lines[i].replace('\n', '')
            if li in SCORE_NAMES:
                prev_scores[li].append(float(eval(lines[i+1])))
                i += 1
            i += 1
    with open(LAST_SUMMARIES_ROUTE, "r") as f:
        prev_scores['summaries'].append(eval(f.read()))
    return prev_scores


def draw_scores(X, Y, scores, experiment_name="Experiment"):
    """Draws several plots that are under the "plots" folder for the
    supplied test values K (X) and b (Y). Also prints out every
    possible summary made in a given experiment result (scores)
    (One for each cluster on each combination of K and b). Resulting output
    is very long, so make sure to specify a '> file.txt' to drop the
    prints there to later user reading.
    It can be used after calling conduct_experiments, and it only does
    the reading on the results saved in that function call.

    Args:
        X (list): Vector of values of K
        Y (list): Vector of values of b
        scores (dict): dict of saved scores during a certain experiment
            (accessed by joblib.load(EXPERIMENT_RESULTS_ROUTE) or similar)
        experiment_name (str, optional): Name given to the experiment.
            Defaults to "Experiment".
    """
    plt.ion()
    X_surface, Y_surface = np.meshgrid(X, Y)
    for first_key in scores.keys():
        if first_key == 'summaries':
            print("||||||||||||||||||||||||||||||||||||||||||||||||||||||")
            print("SUMMARIES OF " + experiment_name)
            index_of_X = 0
            index_of_Y = 0
            for summ in scores[first_key]:
                X_clusters = list(summ.keys())
                Ys = dict()
                no_labels = False
                print("Summaries with K=" +
                      str(X[index_of_X]) + "and b="+str(Y[index_of_Y]))
                for cluster, summ_data in summ.items():
                    print("++++++++++++++++++++++++++++++++++")
                    print("CLASS " + cluster + " summary:")
                    print("----------------------------")
                    print(summ_data[0])
                    print("++++++++++++++++++++++++++++++++++")
                    if summ_data[1] is not None:
                        no_labels = False
                        for k in summ_data[1].keys():
                            Ys[k] = list()
                    else:
                        no_labels = True
                if no_labels:
                    no_labels = False
                    index_of_Y += 1
                    if index_of_Y >= len(Y):
                        index_of_X += 1
                        index_of_Y = 0
                    continue

                for cluster, summ_data in summ.items():
                    for k in Ys.keys():
                        if k not in summ_data[1].keys():
                            Ys[k].append(0.0)
                        else:
                            Ys[k].append(summ_data[1][k])

                pos = -1
                plt.figure()
                for label in Ys.keys():
                    X_axis = np.arange(len(X_clusters))

                    plt.bar(X_axis + pos*0.2, Ys[label], 0.2, label=label)
                    pos += 1

                plt.xticks(X_axis, X_clusters)
                plt.xlabel("Clusters")
                plt.ylabel("Label distribution")
                plt.title("Label distribution in " + experiment_name + "'s " +
                          "clusters with K="+str(X[index_of_X]) +
                          "and b="+str(Y[index_of_Y]))
                plt.legend()

                plt.savefig(os.path.join(os.path.dirname(
                    os.path.abspath(__file__)),
                    "plots/Label distribution_" + experiment_name + "_" +
                    "_with K_"+str(X[index_of_X]) +
                    "_b_"+str(Y[index_of_Y])+'.png'),
                    dpi=300, bbox_inches='tight')
                plt.close()
                index_of_Y += 1
                if index_of_Y >= len(Y):
                    index_of_X += 1
                    index_of_Y = 0

        else:

            plt.figure(figsize=(20, 15))
            ax = plt.axes(projection='3d')  # 3d contour plot
            Z = np.reshape(scores[first_key],
                           (len(try_keywords), len(try_kretain)))

            ax.plot_surface(X_surface, Y_surface, Z, rstride=1,
                            cstride=1, cmap='inferno')  # set labels
            ax.set_xlabel('Keyword count (K)')
            ax.set_ylabel('Keyword retain (b)')
            ax.set_zlabel(first_key)
            plt.title("Score " + experiment_name+": " + first_key)
            ax.view_init(35, 60)

            plt.savefig(os.path.join(os.path.dirname(
                os.path.abspath(__file__)),
                "plots/score_" + experiment_name + "_" + first_key+'.png'),
                dpi=300, bbox_inches='tight')
            plt.close()
    plt.close('all')


try_keywords = np.array([10, 20, 30, 40, 50])
try_kretain = np.array([0.1, 0.25, 0.5, 0.75, 0.9])


def conduct_experiments():
    """Conducts experiments, saving results in several files.
    Use then draw_scores to represent that data in readable form
    """
    scores_test = dict()
    scores_train = dict()
    scores_test_h = dict()
    for key in SCORE_NAMES:
        scores_train[key] = []
        scores_test[key] = []
        scores_test_h[key] = []
    scores_train['summaries'] = []
    scores_test['summaries'] = []
    scores_test_h['summaries'] = []
    for k in try_keywords:
        for kr in try_kretain:
            subprocess.run(["python", os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "train.py"), "--keyw_retain=" +
                str(kr), "--n_keywords="+str(k), "-t=0.4"])
            scores_train = read_last_scores(scores_train)

            subprocess.run(["python", os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "test.py"), "-l"])
            scores_test = read_last_scores(scores_test)

            subprocess.run(["python", os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "test.py"), "--use_hillary",
                "--keyw_retain=" + str(kr), "--n_keywords="+str(k)])
            scores_test_h = read_last_scores(scores_test_h)

    joblib.dump(scores_train, EXPERIMENT_RESULTS_ROUTE)
    joblib.dump(scores_test, EXPERIMENT_RESULTS_ROUTE_TEST)
    joblib.dump(scores_test_h, EXPERIMENT_RESULTS_ROUTE_H_TEST)


if __name__ == "__main__":
    """Conducts experiments or just draws the different plots and summaries
    """
    from argparse import ArgumentParser
    argp = ArgumentParser(description="Conduct experiments")
    argp.add_argument('-r', '--repeat', action='store_true', default=False,
                      help="Repeat experiments. Takes a lot of time (1-2 hours),\
and it may override several autogenerated files, so use it with caution")
    args = argp.parse_args()
    if args.repeat:
        yn = input("Are you sure you want to repeat experiments? [Y/N]")
        if yn == 'Y' or yn == 'y':
            conduct_experiments()
        elif yn == 'N' or yn == 'n':
            pass
        else:
            print("Not recognized input, aborting...")
            exit(0)
    scores_train = joblib.load(EXPERIMENT_RESULTS_ROUTE)
    scores_test = joblib.load(EXPERIMENT_RESULTS_ROUTE_TEST)
    scores_test_h = joblib.load(EXPERIMENT_RESULTS_ROUTE_H_TEST)

    draw_scores(try_keywords, try_kretain, scores_train,
                experiment_name="TRAINING WITH TWEEPY")
    draw_scores(try_keywords, try_kretain, scores_test,
                experiment_name="TESTING WITH TWEEPY")
    draw_scores(try_keywords, try_kretain, scores_test_h,
                experiment_name="TESTING WITH HILLARY")
