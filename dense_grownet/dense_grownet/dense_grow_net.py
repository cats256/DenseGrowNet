import torch
import torch.nn as nn
from .dense_grow_net_base import DenseGrowNetBase


class DenseGrowNet(nn.Module):
    def __init__(self, input_size, output_size, inputs, scaler_class, base_linear_model=None):
        super(DenseGrowNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.scaler_class = scaler_class

        self.scalers = [scaler_class()]
        self.inputs = [self.scalers[0].fit_transform(inputs)]
        self.raw_outputs = []
        self.models = nn.ModuleList()

        if base_linear_model is not None:
            self.models.append(base_linear_model)
        else:
            self.models.append(DenseGrowNetBase(input_size, output_size, is_first_model=True))

    def grow_net(self):
        self.raw_outputs.append(self.models[-1])
        self.models.append(DenseGrowNetBase(self.input_size * (3 ** len(self.models)), self.output_size, is_first_model=False))
        self.scalers.append(self.scaler_class())

    def to_device(self, device):
        self.models.to(device)

    def concatenate_features(self, x):
        with torch.no_grad():
            extracted_features = self.models[-1].extract_features(x, raw_output)
            extracted_features_np = extracted_features.detach().cpu().numpy()

            x_scaled_1 = np.concatenate((x_scaled, extracted_features_np), axis=1)
            x_scaled_1 = x_scaler.fit_transform(x_scaled_1)
