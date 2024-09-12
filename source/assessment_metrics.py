"""
numpy and sklearn version metrics
for torch tensor -> ignite?
"""

import numpy as np
import torch
import ignite.contrib.metrics
import sklearn.metrics
# roc_auc_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
"""
----------------------
Classification metrics 
----------------------
roc
accuracy
sensitivity /recall
specificity
positive predictive value /precision
negative predictive value
MCC 
brier
"""
def roc_auc_binary(y_tru, y_scr):
    """
    :param np.array y_tru : a single dimensional numpy array storing the true values
    :param np.array y_scr:  a numpy array of same dimension of y_tru storing the predicted values by model
    :return: Area under ROC curve for the binary classfication model
    """
    return sklearn.metrics.roc_auc_score(y_tru, y_scr)


def accuracy_np(y_tru, y_scr, divide_by_n=True):
    """
    :param np.array y_tru : a single dimensional numpy array storing the true values
    :param np.array y_scr:  a numpy array of same dimension of y_tru storing the predicted values by model
    :return: Area under ROC curve for the binary classfication model
    """
    total_n = y_tru.shape
    corrects = np.sum(y_scr == y_tru)
    if divide_by_n is True:
        return float(corrects / total_n)
    else:
        return corrects


def accuracy_torch(labels, preds, divide_by_n=True):
    """
    :param torch tensor y_tru : a single dimensional pytorch tensor storing the true values
    :param torch tensor  y_scr:  a pytorch tensor of same dimension of y_tru storing the predicted values by model
    :return: Area under ROC curve for the binary classfication model
    """
    total_n = labels.shape
    corrects = torch.sum(preds == labels.data)
    if divide_by_n is True:
        return float(corrects / total_n)
    else:
        return corrects


"""
----------------------
Regression metrics 
----------------------
Mean Squared Error(MSE)
Root-Mean-Squared-Error(RMSE).
Mean-Absolute-Error(MAE).
R² or Coefficient of Determination.
Adjusted R²"""


def mse_torch(labels, outputs):
    criterion = torch.nn.MSELoss()
    mse = criterion(outputs, labels)
    return mse.item()


def mse_np(labels, outputs, ax=None):
    mse = np.sum(np.square(labels - outputs)).mean(axis=ax)
    return mse


def rsme_torch(labels, outputs):
    rmse = torch.sqrt(mse_torch(outputs, labels))
    return rmse.item()


def mse_np(labels, outputs):
    return np.sqrt(mse_np(labels, outputs))


def mae_torch(labels, outputs):
    criterion = torch.nn.l1_loss()
    mae= criterion(outputs, labels)
    return mae.item()


def mae_np(labels,outputs,ax=None):

    return (np.square(labels - outputs)).mean(axis=ax)

def r_squared_torch(labels,output):
    return ignite.contrib.metrics.regression.R2Score
"""

---------------------
Segmentation/detection metrics 
---------------------
Dice/F1
Jaccard/IoU

"""

