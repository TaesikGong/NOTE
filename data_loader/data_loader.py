from torch.utils.data import DataLoader
import torch
import numpy as np
import time
import math
from .CIFAR10Dataset import CIFAR10Dataset
from .CIFAR100Dataset import CIFAR100Dataset
from .ImageNetDataset import ImageNetDataset

import os
import pickle
import re
import random
import copy

import conf


def keep_order_split(entire_data, train_size, valid_size, test_size):
    # Split the dataset while keeping the original order of it.
    all_indices = [i for i in range(len(entire_data))]

    valid_indices = random.sample(all_indices, valid_size)

    for i in sorted(valid_indices, reverse=True):  # reverse is required
        all_indices.pop(i)

    test_indices = random.sample(all_indices, test_size)

    for i in sorted(test_indices, reverse=True):
        all_indices.pop(i)

    valid_data = torch.utils.data.Subset(entire_data, valid_indices)
    test_data = torch.utils.data.Subset(entire_data, test_indices)
    train_data = torch.utils.data.Subset(entire_data, all_indices)

    return train_data, valid_data, test_data


def split_data(entire_data, valid_split, test_split, train_max_rows, valid_max_rows, test_max_rows):
    valid_size = math.floor(len(entire_data) * valid_split)
    test_size = math.floor(len(entire_data) * test_split)

    train_size = len(entire_data) - valid_size - test_size

    assert (train_size >= 0 and valid_size >= 0 and test_size >= 0)

    train_data, valid_data, test_data = keep_order_split(entire_data, train_size, valid_size, test_size)

    if len(entire_data) > train_max_rows:
        train_data = torch.utils.data.Subset(train_data, range(train_max_rows))
    if len(valid_data) > valid_max_rows:
        valid_data = torch.utils.data.Subset(valid_data, range(valid_max_rows))
    if len(test_data) > test_max_rows:
        test_data = torch.utils.data.Subset(test_data, range(test_max_rows))

    return train_data, valid_data, test_data


def datasets_to_dataloader(datasets, batch_size, concat=True, shuffle=True, drop_last=False):
    if concat:
        data_loader = None
        if len(datasets):
            if type(datasets) is torch.utils.data.dataset.Subset:
                datasets = [datasets]
            if sum([len(dataset) for dataset in datasets]) > 0:  # at least one dataset has data
                data_loader = DataLoader(torch.utils.data.ConcatDataset(datasets), batch_size=batch_size,
                                         shuffle=shuffle, drop_last=drop_last, pin_memory=False)
        return data_loader
    else:
        data_loaders = []
        for dataset in datasets:
            if len(dataset) == 0:
                continue
            else:
                data_loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                drop_last=drop_last, pin_memory=False))

        return data_loaders


def load_cache(dataset, cond, data_file_path, transform=None):
    root = './cached_data/'
    dir = root + str(dataset) + '/'
    filename = re.sub("[^a-zA-Z0-9 \n]", '_', str(cond) + '_' + str(data_file_path))
    if transform:
        filename += '_' + transform + '.pkl'
    else:
        filename += '.pkl'
    cache_path = dir + filename

    if os.path.isfile(cache_path):
        print(f'Cache hit:{cache_path}')
        return torch.load(cache_path)
    else:
        print(f'Cache miss:{cache_path}')
        return None


def save_cache(loaded_data, dataset, cond, data_file_path, transform=None):
    root = './cached_data/'
    dir = root + str(dataset) + '/'
    filename = re.sub("[^a-zA-Z0-9 \n]", '_', str(cond) + '_' + str(data_file_path))
    if transform:
        filename += '_' + transform + '.pkl'
    else:
        filename += '.pkl'
    cache_path = dir + filename
    if not os.path.exists(dir):
        os.makedirs(dir)
    return torch.save(loaded_data, cache_path, pickle_protocol=4)


def domain_data_loader(dataset, domains, file_path, batch_size, train_max_rows=np.inf, valid_max_rows=np.inf,
                       test_max_rows=np.inf, valid_split=0, test_split=0, is_src=True,
                       num_source=9999):
    entire_datasets = []
    train_datasets = []

    valid_datasets = []
    test_datasets = []
    st = time.time()

    if domains is not None:
        if domains == 'src':
            processed_domains = conf.args.opt['src_domains']
        elif isinstance(domains, (list,)):
            processed_domains = domains
        else:
            processed_domains = [domains]
    elif is_src:

        if conf.args.validation:
            processed_domains = sorted(list(set(conf.args.opt['src_domains']) - set([conf.args.tgt]))) # Leave-one-user-out
        else:
            processed_domains = conf.args.opt['src_domains']
    else:
        if conf.args.validation:
            processed_domains = conf.args.opt['src_domains']
        else:
            processed_domains = conf.args.opt['tgt_domains']

    ##-- load dataset per each domain
    print('Domains:{}'.format(processed_domains))


    loaded_data = None

    if dataset in ['cifar10']:

        cond = processed_domains

        transform = 'src' if is_src else 'val'
        loaded_data = load_cache(dataset, processed_domains, file_path, transform=transform)

        if not loaded_data:
            loaded_data = CIFAR10Dataset(file=file_path, domains=cond, max_source=num_source, transform=transform)
            save_cache(loaded_data, dataset, processed_domains, file_path, transform=transform)

    elif dataset in ['cifar100']:

        cond = processed_domains

        transform = 'src' if is_src else 'val'
        loaded_data = load_cache(dataset, processed_domains, file_path, transform=transform)

        if not loaded_data:
            loaded_data = CIFAR100Dataset(file=file_path, domains=cond, max_source=num_source, transform=transform)
            save_cache(loaded_data, dataset, processed_domains, file_path, transform=transform)

    elif dataset in ['imagenet']:

        cond = processed_domains
        transform = 'src' if is_src else 'val'
        loaded_data = load_cache(dataset, processed_domains, file_path, transform=transform)
        #
        if not loaded_data:
            loaded_data = ImageNetDataset(file=file_path, domains=cond, max_source=num_source, transform=transform)
            save_cache(loaded_data, dataset, processed_domains, file_path, transform=transform)


    train_data = loaded_data
    entire_datasets.append(train_data)

    ##-- split each dataset into train, valid, and test
    for train_data in entire_datasets:
        total_len = len(train_data)
        train_data, valid_data, test_data = split_data(train_data, valid_split, test_split, train_max_rows,
                                                       valid_max_rows, test_max_rows)

        train_datasets.append(train_data)
        valid_datasets.append(valid_data)
        test_datasets.append(test_data)

        print('#Multi?:{:d} data_loader len:{:d} Train: {:d}\t# Valid: {:d}\t# Test: {:d}'.format(
            True if domains == ['rest'] else False, total_len, len(train_data), len(valid_data),
            len(test_data)))

    train_datasets = train_datasets[:num_source]
    valid_datasets = valid_datasets[:num_source]
    test_datasets = test_datasets[:num_source]

    print('# Time: {:f} secs'.format(time.time() - st))

    if is_src:
        train_data_loader = datasets_to_dataloader(train_datasets, batch_size=batch_size, concat=True,
                                                   drop_last=True,
                                                   shuffle=True)  # Drop_last for avoding one-sized minibatches for batchnorm layers
    else:
        train_data_loader = datasets_to_dataloader(train_datasets, batch_size=1, concat=True,
                                                   drop_last=False,
                                                   shuffle=False)
    valid_data_loader = datasets_to_dataloader(valid_datasets, batch_size=batch_size, concat=True,
                                               shuffle=False)
    test_data_loader = datasets_to_dataloader(test_datasets, batch_size=batch_size, concat=True, shuffle=False)

    data_loader = {
        'train': train_data_loader,
        'valid': valid_data_loader,
        'test': test_data_loader,
        'num_domains': sum([dataset.dataset.get_num_domains() for dataset in train_datasets]),
    }
    print('num_domains:' + str(data_loader['num_domains']))
    return data_loader


if __name__ == '__main__':
    pass
