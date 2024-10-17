import torch.nn as nn
from .dense_grow_net_base import DenseGrowNetBase


class DenseGrowNet(nn.Module):
    def __init__(self, input_size, output_size, scaler_class, base_linear_model=None):
        super(DenseGrowNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.scaler_class = scaler_class

        self.scalers = [scaler_class()]
        self.models = nn.ModuleList()

        if base_linear_model is not None:
            self.models.append(base_linear_model)
        else:
            self.models.append(DenseGrowNetBase(input_size, output_size, is_first_model=True))

    def grow_net(self):
        self.models.append(DenseGrowNetBase(self.input_size * (3 ** len(self.models)), self.output_size, is_first_model=False))
        self.scalers.append(self.scaler_class())
