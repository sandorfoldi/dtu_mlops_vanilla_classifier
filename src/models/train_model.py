import argparse
import sys
import torch
from torch import nn, optim
from models.model import MyAwesomeModel
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


def main():

    print("Training day and night")
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.1, help='learning rate')
    parser.add_argument('--e', default=5, help='number of epochs')
    parser.add_argument('--x', default='data/processed/train_imgs_tensor.pt')
    parser.add_argument('--y', default='data/processed/train_labels_tensor.pt')
    args = parser.parse_args(sys.argv[2:])

    model = MyAwesomeModel()

    tensor_x = torch.load(args.x)
    tensor_y = torch.load(args.y)

    my_dataset = TensorDataset(tensor_x, tensor_y)
    my_dataloader = DataLoader(my_dataset, batch_size=64)

    criterion = nn.NLLLoss()

    optimizer = optim.SGD(model.parameters(), lr=float(args.lr))

    losses = []

    for e in range(args.e):
        running_loss = 0
        for images, labels in my_dataloader:
            optimizer.zero_grad()

            loss = criterion(model(images), labels)
            # print(loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'loss: {running_loss/len(my_dataloader)}')
        losses.append(running_loss)

    # torch.save(model.state_dict(), 'checkpoint.pth')
    torch.save(model, 'models/checkpoints/model_0.pt')
    plt.plot(losses)
    plt.savefig('reports/figures/fig_0.png')


if __name__ == '__main__':
    main()
