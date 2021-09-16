from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef

def get_performance(true_labels, pred_labels):
    acc = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels, pos_label=1, zero_division=0)
    rec = recall_score(true_labels, pred_labels, pos_label=1, zero_division=0)
    macro_f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    mcc = matthews_corrcoef(true_labels, pred_labels)
    cm = confusion_matrix(true_labels, pred_labels, labels=[1, 0])
    return cm, acc, prec, rec, macro_f1, mcc
