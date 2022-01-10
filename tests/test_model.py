import torch

from src.models.model_0 import MyAwesomeModel

model = MyAwesomeModel()
input_tensor = torch.rand([1, 784])

assert model(input_tensor).shape == torch.Size([1, 10]), 'The output of the model has wrong shape'