import argparse
import os
import sys
import re
import multiprocessing as mp
from tqdm import tqdm
import json
import numpy as np
import collections
from texttable import Texttable




args = None

############# configurable #############
SRC_PREFIX = "reproduce_src"
LOG_PREFIX = "reproduce_methods"
dist_dict = {
    0: 'temporally correlated (non-i.i.d.) test stream',
    1: 'uniformly distributed (i.i.d.) test stream'
}
method_dict = {
    'src': 'Src',
    'bnstats': 'BN_Stats',
    'onda': 'ONDA',
    'pl': 'PseudoLabel',
    'tent': 'TENT',
    'lame': 'LAME',
    'cotta': 'CoTTA',
    'note': 'NOTE',
    'note_iid': 'NOTE*',
}
########################################
def get_avg_online_acc(file_path):
    f = open(file_path)
    json_data = json.load(f)
    f.close()
    return json_data['accuracy'][-1]

def read_dict_json(path):
    tmp_dict = {}
    tmp_dict[path] = get_avg_online_acc(path + '/online_eval.json')
    return tmp_dict

def pretty_print(input_dict, dist, dataset, method_list, seed_list):
    print(f'Classification errors(%) on {dataset.upper()}-C, {dist_dict[dist]}')

    t = Texttable()
    t.set_precision(1)
    if len(seed_list) == 1:
        t_head = ['Method', f'Seed {seed_list[0]}']
    else:
        t_head = ['Method', 'MEAN', 'STDEV']
        for i, seed in enumerate(seed_list):
            t_head.insert(i+1, f'Seed {seed}')
    t.add_row(t_head)

    for method in method_list:
        if dist == 0 and method == 'NOTE*':
            continue
        errors = []
        for seed in seed_list:
            accuracy = input_dict[method][dist][seed]
            error = 100 - accuracy
            errors.append(error)
        if len(seed_list) == 1:
            t_val = [method, errors[0]]
        else:
            t_val = [method]
            t_val.extend(errors)
            t_val.append(np.mean(errors))
            t_val.append(np.std(errors))
        t.add_row(t_val)
    # print table
    print(t.draw())
    print('\n')


def create_acc_dict(dataset, method_list, seed_list):
    tot_dict = collections.defaultdict(lambda : collections.defaultdict(dict))
    for seed in seed_list:
        for dist in dist_dict.keys():
            for method in method_list:
                # print(seed, dist, method)
                pattern_of_path = f'.*{LOG_PREFIX}_'
                avg = ret_avg_acc(seed, dist, method, dataset, pattern_of_path)
                tot_dict[method][dist][seed] = avg
    return tot_dict


def ret_avg_acc(seed, dist, method, dataset, pattern):
    if method == 'Src':
        pattern_of_path = f'.*{method}/.*{LOG_PREFIX}_{seed}.*'
    elif method == 'NOTE*':
        if dist == 0:
            return 0
        pattern_of_path = f'.*NOTE/.*{LOG_PREFIX}_iid_{seed}_dist1_iabn_k4'
    else:
        pattern_of_path = f'.*{method}/{pattern}{seed}.*dist{dist}'
    # print(pattern_of_path)
    root = 'log/' + dataset
    path_list = []
    pattern_of_path = re.compile(pattern_of_path)

    for (path, dir, files) in os.walk(root):
        if pattern_of_path.match(path):
            if not path.endswith('/cp'):  # ignore cp/ dir
                path_list.append(path)
    
    pool = mp.Pool()
    all_dict = {}
    
    with pool as p:
        ret = list(p.imap(read_dict_json, path_list, chunksize=1))
        for d in ret:
            all_dict.update(d)

    avg = 0
    for k, v in sorted(all_dict.items()):
        avg += v
    avg = avg / len(all_dict.keys())
    return avg


def main(args):

    if args.dataset == "all":
        dataset_list = ['cifar10', 'cifar100']
    else:
        dataset_list = [args.dataset]

    if args.method == "all":
        method_list = ['Src', 'BN_Stats', 'ONDA', 'PseudoLabel', 'TENT', 'LAME', 'CoTTA', 'NOTE', 'NOTE*']
    else:
        method_list = [method_dict[args.method]]

    if args.seed == "all":
        seed_list = [0, 1, 2]
    else:
        seed_list = [int(args.seed)]

    print("Processing data logs...")
    for dataset in dataset_list:
        tot_dict = create_acc_dict(dataset, method_list, seed_list)
        for dist in dist_dict.keys():
            pretty_print(tot_dict, dist, dataset, method_list, seed_list)


def parse_arguments(argv):
    """Command line parse."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all',
                        help='dataset used. [cifar10, cifar100, all]')
    parser.add_argument('--method', type=str, default='all',
                        help='method used. [src, bnstats, onda, pl, tent, lame, cotta, note, note_iid, all]')
    parser.add_argument('--seed', type=str, default='all',
                        help='random seed used. [0, 1, 2, all]')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)

