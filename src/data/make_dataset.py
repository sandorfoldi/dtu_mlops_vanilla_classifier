# -*- coding: utf-8 -*-
import glob
import logging
import os
import os.path
import sys
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    train_files = glob.glob(os.path.join(input_filepath, 'train*'))
    test_files = glob.glob(os.path.join(input_filepath, 'test*'))

    train_images = np.concatenate(list(map(lambda l: np.load(l)['images'], train_files)), 0)
    test_images = np.load(test_files[0])['images']

    train_images = train_images.reshape((-1, 784))
    test_images = test_images.reshape((-1, 784))

    train_labels = np.concatenate(list(map(lambda l: np.load(l)['labels'], train_files)), 0)
    test_labels = np.load(test_files[0])['labels']

    train_imgs_tensor = torch.from_numpy(train_images.astype(np.float32))
    train_imgs_tensor = (train_imgs_tensor - train_imgs_tensor.mean()) / train_imgs_tensor.var()

    train_labels_tensor = torch.from_numpy(train_labels.astype(int))

    test_imgs_tensor = torch.from_numpy(test_images.astype(np.float32))
    test_imgs_tensor = (test_imgs_tensor - test_imgs_tensor.mean()) / test_imgs_tensor.var()

    test_labels_tensor = torch.from_numpy(test_labels.astype(int))

    torch.save(train_imgs_tensor, os.path.join(output_filepath, 'train_imgs_tensor.pt'))
    torch.save(train_labels_tensor, os.path.join(output_filepath, 'train_labels_tensor.pt'))
    torch.save(test_imgs_tensor, os.path.join(output_filepath, 'test_imgs_tensor.pt'))
    torch.save(test_labels_tensor, os.path.join(output_filepath, 'test_labels_tensor.pt'))

    print('processed tensors saved')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
