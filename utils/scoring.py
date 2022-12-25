from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import math
import numpy as np

def compute_multiclass_accuracy(y_true, y_pred):
    correct = np.sum(np.array(y_true)==np.array(y_pred))
    acc = correct/float(len(y_true))
    return acc

def compute_precision(y_true, y_pred, pos_class=True):
    true_neg, false_pos, false_neg, true_pos = confusion_matrix(y_true, y_pred).ravel()
    precision = 0.0
    if pos_class:
        if true_pos+false_pos != 0:
            precision = true_pos/(true_pos+false_pos)
    else:
        if true_neg+false_neg != 0:
            precision = true_neg/(true_neg+false_neg)
    return precision

def compute_recall(y_true, y_pred):
    true_neg, false_pos, false_neg, true_pos = confusion_matrix(y_true, y_pred).ravel()
    recall = 0.0
    if true_pos+false_neg != 0:
        recall = true_pos/(true_pos+false_neg)
    return recall

def compute_fscore(y_true, y_pred, beta=1):
    precision = compute_precision(y_true, y_pred)
    recall = compute_recall(y_true, y_pred)
    fscore = 0.0
    if precision+recall!=0:
        fscore = (1+math.pow(beta,2))*(precision*recall)/(math.pow(beta,2)*precision + recall)
    return fscore

def compute_gmean(y_true, y_pred):
    sens = compute_senitivity(y_true, y_pred)
    spec = compute_specificity(y_true, y_pred)
    g_mean = math.sqrt(sens*spec)
    return g_mean

def compute_senitivity(y_true, y_pred):
    true_neg, false_pos, false_neg, true_pos = confusion_matrix(y_true, y_pred).ravel()
    all_pos = true_pos+false_neg
    sens = true_pos/all_pos
    return sens

def compute_specificity(y_true, y_pred):
    true_neg, false_pos, false_neg, true_pos = confusion_matrix(y_true, y_pred).ravel()
    all_neg = true_neg+false_pos
    spec = true_neg/all_neg
    return spec

def compute_micro_acc(y_true, y_pred):
    true_neg, false_pos, false_neg, true_pos = confusion_matrix(y_true, y_pred).ravel()
    sens = compute_senitivity(y_true, y_pred)
    spec = compute_specificity(y_true, y_pred)
    all_pos = true_pos+false_neg
    all_neg = true_neg+false_pos
    all_samples = all_pos + all_neg
    micro_acc = ((all_pos/all_samples)*sens) + ((all_neg/all_samples)*spec)
    return micro_acc

def compute_macro_acc(y_true, y_pred):
    sens = compute_senitivity(y_true, y_pred)
    spec = compute_specificity(y_true, y_pred)
    macro_acc = (sens+spec)/2.0
    return macro_acc

def get_roc_metrics(y_true, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    auc_score = auc(fpr, tpr)
    return auc_score, fpr, tpr

def compute_metrics(y_true, y_pred, y_pred_prob=None):
    auc_score, fpr, tpr = get_roc_metrics(y_true, y_pred_prob)
    conf_matrix_disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    all_metrics = {
        'fpr':fpr,
        'tpr':tpr,
        'auc':auc_score,
        'precision':compute_precision(y_true, y_pred),
        'recall':compute_recall(y_true, y_pred),
        'precision_n':compute_precision(y_true, y_pred,pos_class=False),
        'f1':compute_fscore(y_true, y_pred, beta=1),
        'sensitivity':compute_senitivity(y_true, y_pred),
        'specificity':compute_specificity(y_true, y_pred),
        'macro_acc':compute_macro_acc(y_true, y_pred),
        'micro_acc':compute_micro_acc(y_true, y_pred),
        'g_mean':compute_gmean(y_true, y_pred),
        'confusion_matrix':conf_matrix_disp
    }
    return all_metrics

# score_fn based on gmean to weigh both class equally
def score_fn_gmean(y_true, y_pred):
    g_mean = compute_gmean(y_true, y_pred)
    return g_mean

def score_fn_hybrid(y_true, y_pred):
    g_mean = compute_gmean(y_true, y_pred)
    precision_neg = compute_precision(y_true, y_pred, pos_class=False)
    precision_pos = compute_precision(y_true, y_pred, pos_class=True)
    p_mean = math.sqrt(precision_neg*precision_pos)

    # acc = compute_micro_acc(y_true, y_pred)
    return p_mean + g_mean