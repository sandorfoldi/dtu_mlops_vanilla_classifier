import argparse
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset


def main():
    print("Evaluating until hitting the ceiling")
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--x", default="data/processed/test_imgs_tensor.pt")
    parser.add_argument("--y", default="data/processed/test_labels_tensor.pt")
    parser.add_argument("--m", default="models/model_0.pt")

    args = parser.parse_args(sys.argv[2:])

    model = torch.load(args.m)

    tensor_x = torch.load(args.x)
    tensor_y = torch.load(args.y)

    my_dataset = TensorDataset(tensor_x, tensor_y)
    my_dataloader = DataLoader(my_dataset, batch_size=64)

    criterion = torch.nn.NLLLoss()
    with torch.no_grad():
        # set model to evaluation mode
        model.eval()

        # validation pass here
        running_accuracy = 0
        running_loss = 0
        for images, labels in my_dataloader:
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            running_accuracy += accuracy.item()
            # print(model(images).shape)
            # break
            loss = criterion(model(images), labels)
            running_loss += loss
        print(f'accuracy: {running_accuracy * 100 / len(my_dataloader)}')
        print(f'loss: {running_loss / len(my_dataloader)}')


if __name__ == "__main__":
    main()
