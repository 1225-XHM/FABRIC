import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)


def best_f1_threshold(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denom,
                   out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[np.argmax(f1)])


def evaluate(labels, probabilities, threshold):
    predicted = np.asarray(probabilities) >= threshold
    return {
        "AUROC": float(roc_auc_score(labels, probabilities)),
        "AUPRC": float(average_precision_score(labels, probabilities)),
        "Accuracy": float(accuracy_score(labels, predicted)),
        "Precision": float(precision_score(labels, predicted, zero_division=0)),
        "Recall": float(recall_score(labels, predicted, zero_division=0)),
        "F1": float(f1_score(labels, predicted, zero_division=0)),
        "threshold": float(threshold),
    }
