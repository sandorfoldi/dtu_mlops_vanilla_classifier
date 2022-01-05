import torch
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', default='models/checkpoints/model_0.pt')
    parser.add_argument('--x', default='data/processed/train_imgs_tensor.pt')
    parser.add_argument('--i', default=0)
    args = parser.parse_args(sys.argv[2:])

    model = torch.load(args.m)
    model_alt = torch.nn.Sequential(
        model.l1,
        torch.nn.ReLU(),
        model.l2,
        torch.nn.ReLU(),
    )
    tensor_x = torch.load(args.x)

    inter_repr = model_alt(tensor_x)

    features_2d = TSNE(n_components=2, learning_rate='auto') \
        .fit_transform(inter_repr.detach().numpy())
    plt.plot(features_2d)
    plt.savefig('reports/figures/fig_inter_repr_0.png')


if __name__ == '__main__':
    main()
