from sklearn.metrics import f1_score, recall_score, precision_score

def metrics(all_golds, all_preds, labels =  None):
    p = precision_score(all_golds, all_preds, labels=labels, zero_division=0, average='micro')
    r = recall_score(all_golds, all_preds, labels=labels, zero_division=0, average='micro')
    f = f1_score(all_golds, all_preds, labels=labels, zero_division=0, average='micro')
    return {'p': p * 100, 'r': r * 100, 'f': f * 100}