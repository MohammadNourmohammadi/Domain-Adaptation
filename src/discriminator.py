import torch
from torch.autograd import Function
from torch.nn import Module, Sequential, Linear, ReLU

class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DomainDiscriminator(Module):
    def __init__(self, in_features, hidden_features):
        super(DomainDiscriminator, self).__init__()
        self.discriminator = Sequential(
            Linear(in_features, hidden_features),
            ReLU(inplace=True),
            Linear(hidden_features, 1)
        )
        self.grl = GradientReversalLayer.apply

    def forward(self, x, alpha=1.0):
        x_reversed = self.grl(x, alpha)
        return self.discriminator(x_reversed).sigmoid()