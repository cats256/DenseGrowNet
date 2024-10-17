import torch
import torch.nn as nn


class CustomLinearLayer(nn.Module):
    def __init__(self, input_size, output_size, init="looks_linear"):
        super(CustomLinearLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=True)
        nn.init.zeros_(self.linear.bias)

        if init == "zero":
            nn.init.zeros_(self.linear.weight)
        elif init == "looks_linear":
            if input_size * 2 != output_size:
                raise ValueError("Output size must be twice that of input size")

            with torch.no_grad():
                weight = torch.zeros(input_size * 2, input_size)

                for i in range(self.linear.in_features):
                    weight[2 * i, i] = 1
                    weight[2 * i + 1, i] = -1

                self.linear.weight.copy_(weight)
                nn.init.zeros_(self.linear.bias)

            """ Example matrix: [
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1]
            ] """

    def forward(self, x):
        return self.linear(x)
