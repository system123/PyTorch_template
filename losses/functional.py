import torch
from torch import nn

def custom_loss(y_pred, y):
    return(0)

def cross_entropy_one_hot(input, target):
    _, labels = torch.max(target, dim=1)
    return nn.CrossEntropyLoss()(input, labels)
