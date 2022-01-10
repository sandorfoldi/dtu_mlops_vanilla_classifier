import os

import pytest
import torch


data_path = 'data/processed/train_imgs_tensor.pt'


@pytest.mark.skipif(not os.path.exists(data_path), reason="Data files not found")
def test_data_processed_shape():
    tensor_x = torch.load('data/processed/train_imgs_tensor.pt')
    tensor_y = torch.load('data/processed/train_labels_tensor.pt')

    assert tensor_x.shape == torch.Size([40000, 784]), 'train_img has wrong shape:('
    assert tensor_y.shape == torch.Size([40000]), 'train_labels has wrong shape:('
