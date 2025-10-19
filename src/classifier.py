from torch.nn import Module, Linear

class NodeClassifier(Module):
    def __init__(self, in_features, num_classes):
        super(NodeClassifier, self).__init__()
        self.fc = Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)