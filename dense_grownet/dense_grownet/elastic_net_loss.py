import torch.nn as nn


class ElasticNetLoss(nn.Module):
    def __init__(self, criterion, l1_lambda, l2_lambda):
        super(ElasticNetLoss, self).__init__()
        self.criterion = criterion
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, outputs, labels, model):
        loss = self.criterion(outputs, labels)

        l1_norm = sum(p.abs().sum() for name, p in model.named_parameters() if "bias" not in name)
        l2_norm = sum(p.pow(2.0).sum() for name, p in model.named_parameters() if "bias" not in name)

        loss += self.l1_lambda * l1_norm + self.l2_lambda * l2_norm
        return loss
