import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve,roc_auc_score

def bootstrap(y_tru, y_scr, filepath, n_bootstraps=2000, randomseed=245):
    """

    :param y_tru: true label stored as numpy array
    :param y_scr: predicted score stored as numpy array
    :param n_bootstraps:
    :param randomseed:
    :return:
    """

    # original_auc=roc_auc_score(y_tru, y_scr)
    if type(y_tru) != 'numpy.ndarray' or type(y_scr) != 'numpy.ndarray':
        try:
            y_tru, y_scr = np.array(y_tru), np.array(y_scr)
        except:
            print("unable to convert inputs to numpy arrays, check input")
            return
    total = len(y_tru)
    original_acc = (y_scr == y_tru).sum() / total
    print("Original Accuracy: {:0.1f}".format(original_acc * 100))

    custom_seed = randomseed  # control reproducibility
    bootstrapped_scores = []

    rng = np.random.RandomState(custom_seed)

    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_scr) - 1, len(y_scr))
        # indices=np.random.choice( range(len(y_pred) - 1), len(y_pred), replace=True)  # this also with replacement
        if len(np.unique(y_tru[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        bootstrap_acc = ((y_tru[indices] == y_scr[indices]).sum()) / total
        bootstrapped_scores.append(bootstrap_acc)
        # print("Bootstrap #{} ACC: {:0.2f}".format(i + 1, bootstrap_acc))

    # plot bootstrapping results

    plt.hist(bootstrapped_scores, bins=32, color="darkcyan", ec="darkslategray")
    plt.title('Histogram of the bootstrapped accuracies on {} bootstrapping sample'.format(n_bootstraps))
    fig_outpath = filepath + 'acc_bootstrap_hist.png'
    plt.savefig(fig_outpath)
    plt.close()
    # obtain the 95 % CI from the results
    sorted_accuracies = np.array(bootstrapped_scores)
    sorted_accuracies.sort()
    conf_low = sorted_accuracies[int(0.025 * len(sorted_accuracies))]
    conf_up = sorted_accuracies[int(0.975 * len(sorted_accuracies))]
    print(
        'ACC with 95% confidence interval the ACC : {:.1f} ({:.1f} - {:.1f})'.format(original_acc * 100, conf_low * 100,
                                                                                     conf_up * 100))
    return original_acc, conf_low, conf_up

def bootstrap_auc(y_tru, y_scr, n_bootstraps=2000, randomseed=245):
    """

    :param y_tru: true label stored as numpy array
    :param y_scr: predicted score stored as numpy array
    :param n_bootstraps:
    :param randomseed:
    :return:
    """

    # original_auc=roc_auc_score(y_tru, y_scr)
    if type(y_tru) != 'numpy.ndarray' or type(y_scr) != 'numpy.ndarray':
        try:
            y_tru, y_scr = np.array(y_tru), np.array(y_scr)
        except:
            print("unable to convert inputs to numpy arrays, check input")
            return
    total = len(y_tru)
    original_auc = roc_auc_score(y_tru,y_scr)
    print("Original AUC: {:0.3f}".format(original_auc ))

    custom_seed = randomseed  # control reproducibility
    bootstrapped_scores = []

    rng = np.random.RandomState(custom_seed)

    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_scr) - 1, len(y_scr))
        # indices=np.random.choice( range(len(y_pred) - 1), len(y_pred), replace=True)  # this also with replacement
        if len(np.unique(y_tru[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        bootstrap_auc = roc_auc_score(y_tru[indices], y_scr[indices])

        bootstrapped_scores.append(bootstrap_auc)
        # print("Bootstrap #{} ACC: {:0.2f}".format(i + 1, bootstrap_acc))

    # plot bootstrapping results
    # obtain the 95 % CI from the results
    sorted_auc = np.array(bootstrapped_scores)
    sorted_auc.sort()
    conf_low = sorted_auc[int(0.025 * len(sorted_auc))]
    conf_up = sorted_auc[int(0.975 * len(sorted_auc))]
    print(
        'AUC with 95% confidence interval: {:.3f} ({:.3f} - {:.3f})'.format(original_auc,
                                                                                     conf_low,
                                                                                     conf_up))
    return original_auc, conf_low, conf_up

