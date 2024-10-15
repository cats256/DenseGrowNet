import torch
import torch.nn as nn
from .custom_linear_layer import CustomLinearLayer


class DenseGrowNet(nn.Module):
    def __init__(self, input_size, output_size, is_first_model):
        super(DenseGrowNet, self).__init__()
        self.activation = nn.ReLU()

        self.input_size = input_size
        self.output_size = output_size

        self.is_first_model = is_first_model

        if is_first_model:
            self.first_layer = CustomLinearLayer(input_size, output_size, init="zero")
        else:
            self.first_layer = CustomLinearLayer(input_size, input_size * 2, init="looks_linear")
            self.last_layer = CustomLinearLayer(input_size * 2, output_size, init="zero")

    def forward(self, x, prev_output=None):
        if self.is_first_model and prev_output is not None:
            raise ValueError("This is the first model and prev_output is passed in")
        if not self.is_first_model and prev_output is None:
            raise ValueError("This is not the first model and prev_output is not passed in")

        if prev_output is None:
            return self.first_layer(x)

        x = self.activation(self.first_layer(x))
        return self.last_layer(x) + prev_output

    def extract_features(self, x, prev_output=None):
        if self.is_first_model:
            raise ValueError("This is not intended for the first model")
        if not self.is_first_model and prev_output is None:
            raise ValueError("This is not the first model and prev_output is not passed in")

        return self.activation(self.first_layer(x))
