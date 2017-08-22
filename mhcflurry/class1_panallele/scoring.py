from __future__ import (
    print_function,
    division,
    absolute_import,
)
import sklearn
import numpy
import scipy

import mhcflurry


def make_scores(
        ic50_y,
        ic50_y_pred,
        sample_weight=None,
        threshold_nm=0.5,
        max_ic50=50000):
    """
    Calculate AUC, F1, and Kendall Tau scores.

    Parameters
    -----------
    ic50_y : float list
        true IC50s (i.e. affinities)

    ic50_y_pred : float list
        predicted IC50s

    sample_weight : float list [optional]

    threshold_nm : float [optional]

    max_ic50 : float [optional]

    Returns
    -----------
    dict with entries "auc", "f1", "tau"
    """

    y_pred = mhcflurry.regression_target.ic50_to_regression_target(
        ic50_y_pred, max_ic50)
    try:
        auc = sklearn.metrics.roc_auc_score(
            ic50_y >= threshold_nm,
            y_pred,
            sample_weight=sample_weight)
    except ValueError:
        auc = numpy.nan
    try:
        f1 = sklearn.metrics.f1_score(
            ic50_y >= threshold_nm,
            ic50_y_pred >= threshold_nm,
            sample_weight=sample_weight)
    except ValueError:
        f1 = numpy.nan
    try:
        tau = scipy.stats.kendalltau(ic50_y_pred, ic50_y)[0]
    except ValueError:
        tau = numpy.nan
    try:
        acc = sklearn.metrics.accuracy_score(
            ic50_y >= threshold_nm,
            y_pred >= threshold_nm,
            sample_weight=sample_weight)
    except ValueError:
        acc = numpy.nan

    return dict(
        auc=auc,
        acc=acc,
        f1=f1,
        tau=tau)

def classification_report(
        ic50_y,
        ic50_y_pred,
        sample_weight=None,
        threshold = 0.5
        ):
    """
    Return a text report showing the main classification metrics.

    Parameters
    -----------
    ic50_y : int list
        elution property (i.e. true/false)

    ic50_y_pred : float list
        predicted elution property

    sample_weight : float list [optional]

    threshold : float [optional]

    Returns
    -----------
    print the report table 
    """
    labels = [0, 1]
    try:
        print(sklearn.metrics.classification_report(
            ic50_y >= threshold,
            ic50_y_pred >= threshold,
            labels = labels,
            sample_weight=sample_weight))
    except ValueError:
        print("Cannot generate classification report")

