from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 10)

    def forward(self, x):
        if x.shape[1] != 784:
            raise ValueError('hababla')
        x = self.l1(x)
        x = nn.ReLU()(x)
        x = self.l2(x)
        x = nn.ReLU()(x)
        x = self.l3(x)
        x = nn.LogSoftmax(dim=1)(x)
        return x
