import torch
import pytest
import os.path

from src.models.model_0 import MyAwesomeModel
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

data_path = 'data/processed/train_imgs_tensor.pt'

model = MyAwesomeModel()

@pytest.mark.skipif(not os.path.exists(data_path), reason="Data files not found")
def test_error_on_wrong_shape():
    with pytest.raises(ValueError, match='hababla'):
        model(torch.rand((1,2)))

@pytest.mark.parametrize("shape", [(1, 784), (16, 784), (64, 784)])
def test_error_on_different_input_shapes(shape):
    model(torch.rand(shape))