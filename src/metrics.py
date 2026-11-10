import torch

def accuracy(y_true, y_pred):
    return (y_true == y_pred).float().mean().item()          # sahi predictions ka hissa

def confusion_matrix(y_true, y_pred, k=5):
    C = torch.zeros(k, k, dtype=torch.long)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        C[t, p] += 1                                         # row = asal class, column = prediction
    return C

def per_class_accuracy(C):
    return C.diag() / C.sum(dim=1)                           # C_jj / (row j ka jod)