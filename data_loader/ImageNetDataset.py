import os
import warnings
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, ImageFolder

from PIL import Image

import pandas as pd
import time
import numpy as np
import sys
import conf
import json
import tqdm as tqdm

opt = conf.ImageNetOpt




class ImageNetDataset(torch.utils.data.Dataset):

    def __init__(self, file='../dataset/ichar/minmax_scaling_all.csv',
                 domains=None, activities=None,
                 max_source=100, transform='none'):
        st = time.time()
        self.domains = domains
        self.activity = activities
        self.max_source = max_source

        self.domains = domains
        self.features = None
        self.class_labels = None
        self.domain_labels = None
        self.file_path = opt['file_path']
        self.transform_type = transform

        assert (len(domains) > 0)
        if domains[0].startswith('original'):
            self.path = 'origin/Data/CLS-LOC/train/'
        elif domains[0].startswith('test'):
            self.path = 'origin/Data/CLS-LOC/val/'
        else :
            self.path = 'corrupted/'
            corruption, severity =domains[0].split('-')
            self.path += corruption+'/'+severity+'/'

        if transform == 'src':
            self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        elif transform == 'val':
            self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        else:
            raise NotImplementedError

        self.preprocessing()

    def preprocessing(self):

        path = self.file_path+'/'+self.path
        self.features = []
        self.class_labels = []
        self.domain_labels = []
        print('preprocessing images..')
        self.dataset = ImageFolder(path, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def get_num_domains(self):
        return len(self.domains)

    def get_datasets_per_domain(self):
        return self.datasets

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        img, cl = self.dataset[idx]
        return img, cl, 0


if __name__ == '__main__':
    pass
