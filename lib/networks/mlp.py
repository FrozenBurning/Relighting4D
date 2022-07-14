import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, widths, act=None, skip_at=None):
        super(MLP, self).__init__()
        depth = len(widths)
        self.input_dim = input_dim

        if act is None:
            act = [None] * depth
        assert len(act) == depth
        self.layers = nn.ModuleList()
        self.activ = None
        prev_w = self.input_dim
        i = 0
        for w, a in zip(widths, act):
            if isinstance(a, str):
                if a == 'relu':
                    self.activ = nn.ReLU()
                elif a == 'softplus':
                    self.activ = nn.Softplus()
                elif a == 'sigmoid':
                    self.activ = nn.Sigmoid()
                else:
                    raise NotImplementedError
            layer = nn.Linear(prev_w, w)
            prev_w = w
            if skip_at and i in skip_at:
                prev_w += input_dim
            self.layers.append(layer)
            i += 1
        self.skip_at = skip_at

    def forward(self, x):
        x_ = x + 0
        for i, layer in enumerate(self.layers):
            # print(i)
            # print(x_.shape)
            if self.activ:
                y = self.activ(layer(x_))
            else:
                y = layer(x_)
            if self.skip_at and i in self.skip_at:
                y = torch.cat((y, x), dim=-1)
            x_ = y
            # print(y.shape)
        return y
