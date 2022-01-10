import argparse
import glob

import cv2
import numpy as np
import torch


def main():
    print("Prediction running")
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("x", help="folder or pt file")
    parser.add_argument("m", help=" i.e. models/model_0.pt")

    args = parser.parse_args()

    model = torch.load(args.m)
    if args.x[-3:] == '.pt':
        tensor_x = torch.load(args.x)

        ps = torch.exp(model(tensor_x))
        top_p, top_class = ps.topk(1, dim=1)

        print(top_class)
    else:
        img_paths = glob.glob(args.x+'/*.png')
        print(len(img_paths))
        img_list = []
        for img_path in img_paths:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (28, 28))

            img = np.array(img, dtype=np.float32)
            img = (img - img.mean()) / img.var()

            img_list.append(img.reshape(1, -1))
        img_arr = np.concatenate(img_list, 0)
        tensor_x = torch.from_numpy(img_arr)
        print(tensor_x)

        ps = torch.exp(model(tensor_x))
        top_p, top_class = ps.topk(1, dim=1)

        print(top_class)


if __name__ == "__main__":
    main()
