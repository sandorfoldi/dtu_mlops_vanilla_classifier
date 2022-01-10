import torch


tensor_x = torch.load('data/processed/train_imgs_tensor.pt')
tensor_y = torch.load('data/processed/train_labels_tensor.pt')

assert tensor_x.shape == torch.Size([40000, 784]), 'train_img has wrong shape:('
assert tensor_y.shape == torch.Size([40000]), 'train_labels has wrong shape:('

