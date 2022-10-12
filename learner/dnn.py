import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import math
import conf
import random
import pandas as pd
from torch.utils.data import DataLoader
from utils import memory

from utils import iabn
from utils.logging import *
from utils.normalize_layer import *
device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(
    conf.args.gpu_idx)  # this prevents unnecessary gpu memory allocation to cuda:0 when using estimator


class DNN():
    def __init__(self, model, source_dataloader, target_dataloader, write_path):
        self.device = device

        # init dataloader
        self.source_dataloader = source_dataloader

        self.target_dataloader = target_dataloader

        if conf.args.dataset in ['cifar10', 'cifar100'] and conf.args.tgt_train_dist == 0:
            self.tgt_train_dist = 4  # Dirichlet is default for non-real-distribution data
        else:
            self.tgt_train_dist = conf.args.tgt_train_dist
        self.target_data_processing()

        self.write_path = write_path

        ################## Init & prepare model###################
        self.conf_list = []

        # Load model
        if conf.args.model in ['wideresnet28-10', 'resnext29']:
            self.net = model
        elif 'resnet' in conf.args.model:
            num_feats = model.fc.in_features
            model.fc = nn.Linear(num_feats, conf.args.opt['num_class']) # match class number
            self.net = model
        else:
            self.net = model.Net()


        # IABN
        if conf.args.iabn:
            iabn.convert_iabn(self.net)

        if conf.args.load_checkpoint_path and conf.args.model not in ['wideresnet28-10', 'resnext29']:  # false if conf.args.load_checkpoint_path==''
            self.load_checkpoint(conf.args.load_checkpoint_path)

        # Add normalization layers
        norm_layer = get_normalize_layer(conf.args.dataset)
        if norm_layer:
            self.net = torch.nn.Sequential(norm_layer, self.net)

        self.net.to(device)


        ##########################################################





        # init criterions, optimizers, scheduler
        if conf.args.method == 'Src':
            if conf.args.dataset in ['cifar10', 'cifar100', 'harth', 'reallifehar', 'extrasensory']:
                self.optimizer = torch.optim.SGD(
                                  self.net.parameters(),
                                  conf.args.opt['learning_rate'],
                                  momentum=conf.args.opt['momentum'],
                                  weight_decay=conf.args.opt['weight_decay'],
                                  nesterov=True)

                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=conf.args.epoch * len(self.source_dataloader['train']))
            elif conf.args.dataset in ['tinyimagenet']:
                    self.optimizer = torch.optim.SGD(
                        self.net.parameters(),
                        conf.args.opt['learning_rate'],
                        momentum=conf.args.opt['momentum'],
                        weight_decay=conf.args.opt['weight_decay'],
                        nesterov=True)
            else:
                self.optimizer = optim.Adam(self.net.parameters(), lr=conf.args.opt['learning_rate'],
                                            weight_decay=conf.args.opt['weight_decay'])
        else:
            self.optimizer = optim.Adam(self.net.parameters(), lr=conf.args.opt['learning_rate'],
                                    weight_decay=conf.args.opt['weight_decay'])

        self.class_criterion = nn.CrossEntropyLoss()

        # online learning
        if conf.args.memory_type == 'FIFO':
            self.mem = memory.FIFO(capacity=conf.args.memory_size)
        elif conf.args.memory_type == 'Reservoir':
            self.mem = memory.Reservoir(capacity=conf.args.memory_size)
        elif conf.args.memory_type == 'PBRS':
            self.mem = memory.PBRS(capacity=conf.args.memory_size)

        self.json = {}
        self.l2_distance = []
        self.occurred_class = [0 for i in range(conf.args.opt['num_class'])]


    def target_data_processing(self):

        features = []
        cl_labels = []
        do_labels = []

        for b_i, (feat, cl, dl) in enumerate(self.target_dataloader['train']):#must be loaded from dataloader, due to transform in the __getitem__()
            features.append(feat.squeeze(0))# batch size is 1
            cl_labels.append(cl.squeeze())
            do_labels.append(dl.squeeze())

        tmp = list(zip(features, cl_labels, do_labels))
        # for _ in range(
        #         conf.args.nsample):  # this will make more diverse training samples under a fixed seed, when rand_nsample==True. Otherwise, it will just select first k samples always
        #     random.shuffle(tmp)

        features, cl_labels, do_labels = zip(*tmp)
        features, cl_labels, do_labels = list(features), list(cl_labels), list(do_labels)

        num_class = conf.args.opt['num_class']

        result_feats = []
        result_cl_labels = []
        result_do_labels = []

        # real distribution
        if self.tgt_train_dist == 0:
            num_samples = conf.args.nsample if conf.args.nsample < len(features) else len(features)
            for _ in range(num_samples):
                tgt_idx = 0
                result_feats.append(features.pop(tgt_idx))
                result_cl_labels.append(cl_labels.pop(tgt_idx))
                result_do_labels.append(do_labels.pop(tgt_idx))

        # random distribution
        if self.tgt_train_dist == 1:
            num_samples = conf.args.nsample if conf.args.nsample < len(features) else len(features)
            for _ in range(num_samples):
                tgt_idx = np.random.randint(len(features))
                result_feats.append(features.pop(tgt_idx))
                result_cl_labels.append(cl_labels.pop(tgt_idx))
                result_do_labels.append(do_labels.pop(tgt_idx))

        # dirichlet distribution
        elif self.tgt_train_dist == 4:
            dirichlet_numchunks = conf.args.opt['num_class']

            # https://github.com/IBM/probabilistic-federated-neural-matching/blob/f44cf4281944fae46cdce1b8bc7cde3e7c44bd70/experiment.py
            min_size = -1
            N = len(features)
            min_size_thresh = 10 #if conf.args.dataset in ['tinyimagenet'] else 10
            while min_size < min_size_thresh:  # prevent any chunk having too less data
                idx_batch = [[] for _ in range(dirichlet_numchunks)]
                idx_batch_cls = [[] for _ in range(dirichlet_numchunks)] # contains data per each class
                for k in range(num_class):
                    cl_labels_np = torch.Tensor(cl_labels).numpy()
                    idx_k = np.where(cl_labels_np == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(
                        np.repeat(conf.args.dirichlet_beta, dirichlet_numchunks))

                    # balance
                    proportions = np.array([p * (len(idx_j) < N / dirichlet_numchunks) for p, idx_j in
                                            zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

                    # store class-wise data
                    for idx_j, idx in zip(idx_batch_cls, np.split(idx_k, proportions)):
                        idx_j.append(idx)

            sequence_stats = []

            # create temporally correlated toy dataset by shuffling classes
            for chunk in idx_batch_cls:
                cls_seq = list(range(num_class))
                np.random.shuffle(cls_seq)
                for cls in cls_seq:
                    idx = chunk[cls]
                    result_feats.extend([features[i] for i in idx])
                    result_cl_labels.extend([cl_labels[i] for i in idx])
                    result_do_labels.extend([do_labels[i] for i in idx])
                    sequence_stats.extend(list(np.repeat(cls, len(idx))))

            # trim data if num_sample is smaller than the original data size
            num_samples = conf.args.nsample if conf.args.nsample < len(result_feats) else len(result_feats)
            result_feats = result_feats[:num_samples]
            result_cl_labels = result_cl_labels[:num_samples]
            result_do_labels = result_do_labels[:num_samples]

        remainder = len(result_feats) % conf.args.update_every_x  # drop leftover samples
        if remainder == 0:
            pass
        else:
            result_feats = result_feats[:-remainder]
            result_cl_labels = result_cl_labels[:-remainder]
            result_do_labels = result_do_labels[:-remainder]

        try:
            self.target_train_set = (torch.stack(result_feats),
                                     torch.stack(result_cl_labels),
                                     torch.stack(result_do_labels))
        except:
            self.target_train_set = (torch.stack(result_feats),
                                     result_cl_labels,
                                     torch.stack(result_do_labels))
    def save_checkpoint(self, epoch, epoch_acc, best_acc, checkpoint_path):
        if isinstance(self.net, nn.Sequential):
            if isinstance(self.net[0],NormalizeLayer):
                cp = self.net[1]
        else:
            cp = self.net

        torch.save(cp.state_dict(), checkpoint_path)


    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{conf.args.gpu_idx}')
        self.net.load_state_dict(checkpoint, strict=True)
        self.net.to(device)

    def get_loss_and_confusion_matrix(self, classifier, criterion, data, label):
        preds_of_data = classifier(data)

        labels = [i for i in range(len(conf.args.opt['classes']))]

        loss_of_data = criterion(preds_of_data, label)
        pred_label = preds_of_data.max(1, keepdim=False)[1]
        cm = confusion_matrix(label.cpu(), pred_label.cpu(), labels=labels)
        return loss_of_data, cm, preds_of_data

    def get_loss_cm_error(self, classifier, criterion, data, label):
        preds_of_data = classifier(data)
        labels = [i for i in range(len(conf.args.opt['classes']))]

        loss_of_data = criterion(preds_of_data, label)
        pred_label = preds_of_data.max(1, keepdim=False)[1]
        assert (len(label) == len(pred_label))
        cm = confusion_matrix(label.cpu(), pred_label.cpu(), labels=labels)
        errors = [0 if label[i] == pred_label[i] else 1 for i in range(len(label))]
        return loss_of_data, cm, errors

    def log_loss_results(self, condition, epoch, loss_avg):

        if condition == 'train_online':
            # print loss
            print('{:s}: [current_sample: {:d}]'.format(
                condition, epoch
            ))
        else:
            # print loss
            print('{:s}: [epoch: {:d}]\tLoss: {:.6f} \t'.format(
                condition, epoch, loss_avg
            ))

        return loss_avg

    def log_accuracy_results(self, condition, suffix, epoch, cm_class):

        assert (condition in ['valid', 'test'])
        # assert (suffix in ['labeled', 'unlabeled', 'test'])

        class_accuracy = 100.0 * np.sum(np.diagonal(cm_class)) / np.sum(cm_class)

        print('[epoch:{:d}] {:s} {:s} class acc: {:.3f}'.format(epoch, condition, suffix, class_accuracy))

        return class_accuracy

    def train(self, epoch):
        """
        Train the model
        """

        # setup models

        self.net.train()

        class_loss_sum = 0.0

        total_iter = 0

        if conf.args.method in ['Src', 'Src_Tgt']:
            num_iter = len(self.source_dataloader['train'])
            total_iter += num_iter

            for batch_idx, labeled_data in tqdm(enumerate(self.source_dataloader['train']), total=num_iter):
                feats, cls, _ = labeled_data
                feats, cls = feats.to(device), cls.to(device)

                if torch.isnan(feats).any() or torch.isinf(feats).any(): # For reallifehar debugging
                    print('invalid input detected at iteration ', batch_idx)
                    exit(1)
                # compute the feature
                preds = self.net(feats)
                if torch.isnan(preds).any() or torch.isinf(preds).any(): # For reallifehar debugging
                    print('invalid input detected at iteration ', batch_idx)
                    exit(1)
                class_loss = self.class_criterion(preds, cls)
                class_loss_sum += float(class_loss * feats.size(0))

                if torch.isnan(class_loss).any() or torch.isinf(class_loss).any(): # For reallifehar debugging
                    print('invalid input detected at iteration ', batch_idx)
                    exit(1)


                self.optimizer.zero_grad()
                class_loss.backward()
                self.optimizer.step()
                if conf.args.dataset in ['cifar10', 'cifar100', 'harth', 'reallifehar', 'extrasensory']:
                    self.scheduler.step()

        ######################## LOGGING #######################

        self.log_loss_results('train', epoch=epoch, loss_avg=class_loss_sum / total_iter)
        avg_loss = class_loss_sum / total_iter
        return avg_loss

    def logger(self, name, value, epoch, condition):

        if not hasattr(self, name + '_log'):
            exec(f'self.{name}_log = []')
            exec(f'self.{name}_file = open(self.write_path + name + ".txt", "w")')

        exec(f'self.{name}_log.append(value)')

        if isinstance(value, torch.Tensor):
            value = value.item()
        write_string = f'{epoch}\t{value}\n'
        exec(f'self.{name}_file.write(write_string)')

    def evaluation(self, epoch, condition):
        # Evaluate with a batch of samples, which is a typical way of evaluation. Used for pre-training or offline eval.

        self.net.eval()

        with torch.no_grad():
            inputs, cls, dls = self.target_train_set
            tgt_inputs = inputs.to(device)
            tgt_cls = cls.to(device)

            preds = self.net(tgt_inputs)

            labels = [i for i in range(len(conf.args.opt['classes']))]

            class_loss_of_test_data = self.class_criterion(preds, tgt_cls)
            y_pred = preds.max(1, keepdim=False)[1]
            class_cm_test_data = confusion_matrix(tgt_cls.cpu(), y_pred.cpu(), labels=labels)


        print('{:s}: [epoch : {:d}]\tLoss: {:.6f} \t'.format(
            condition, epoch, class_loss_of_test_data
        ))
        class_accuracy = 100.0 * np.sum(np.diagonal(class_cm_test_data)) / np.sum(class_cm_test_data)
        print('[epoch:{:d}] {:s} {:s} class acc: {:.3f}'.format(epoch, condition, 'test', class_accuracy))


        self.logger('accuracy', class_accuracy, epoch, condition)
        self.logger('loss', class_loss_of_test_data, epoch, condition)

        return class_accuracy, class_loss_of_test_data, class_cm_test_data

    def evaluation_online(self, epoch, condition, current_samples):
        # Evaluate with online samples that come one by one while keeping the order.

        self.net.eval()

        with torch.no_grad():

            # extract each from list of current_sample
            features, cl_labels, do_labels = current_samples


            feats, cls, dls = (torch.stack(features), torch.stack(cl_labels), torch.stack(do_labels))
            feats, cls, dls = feats.to(device), cls.to(device), dls.to(device)

            if conf.args.method == 'LAME':
                y_pred = self.batch_evaluation(feats).argmax(-1)

            elif conf.args.method == 'CoTTA':
                x = feats
                anchor_prob = torch.nn.functional.softmax(self.net_anchor(x), dim=1).max(1)[0]
                standard_ema = self.net_ema(x)

                N = 32
                outputs_emas = []

                # Threshold choice discussed in supplementary
                # enable data augmentation for vision datasets
                if anchor_prob.mean(0) < self.ap:
                    for i in range(N):
                        outputs_ = self.net_ema(self.transform(x)).detach()
                        outputs_emas.append(outputs_)
                    outputs_ema = torch.stack(outputs_emas).mean(0)
                else:
                    outputs_ema = standard_ema
                y_pred=outputs_ema
                y_pred = y_pred.max(1, keepdim=False)[1]

            else:

                y_pred = self.net(feats)
                y_pred = y_pred.max(1, keepdim=False)[1]

            ###################### SAVE RESULT
            # get lists from json

            try:
                true_cls_list = self.json['gt']
                pred_cls_list = self.json['pred']
                accuracy_list = self.json['accuracy']
                f1_macro_list = self.json['f1_macro']
                distance_l2_list = self.json['distance_l2']
            except KeyError:
                true_cls_list = []
                pred_cls_list = []
                accuracy_list = []
                f1_macro_list = []
                distance_l2_list = []

            # append values to lists
            true_cls_list += [int(c) for c in cl_labels]
            pred_cls_list += [int(c) for c in y_pred.tolist()]
            cumul_accuracy = sum(1 for gt, pred in zip(true_cls_list, pred_cls_list) if gt == pred) / float(
                len(true_cls_list)) * 100
            accuracy_list.append(cumul_accuracy)
            f1_macro_list.append(f1_score(true_cls_list, pred_cls_list,
                                          average='macro'))

            self.occurred_class = [0 for i in range(conf.args.opt['num_class'])]

            # epoch: 1~len(self.target_train_set[0])
            progress_checkpoint = [int(i * (len(self.target_train_set[0]) / 100.0)) for i in range(1, 101)]
            for i in range(epoch + 1 - len(current_samples[0]), epoch + 1):  # consider a batch input
                if i in progress_checkpoint:
                    print(
                        f'[Online Eval][NumSample:{i}][Epoch:{progress_checkpoint.index(i) + 1}][Accuracy:{cumul_accuracy}]')

            # update self.json file
            self.json = {
                'gt': true_cls_list,
                'pred': pred_cls_list,
                'accuracy': accuracy_list,
                'f1_macro': f1_macro_list,
                'distance_l2': distance_l2_list,
            }


    def dump_eval_online_result(self, is_train_offline=False):

        if is_train_offline:

            feats, cls, dls = self.target_train_set

            for num_sample in range(0, len(feats), conf.args.opt['batch_size']):
                current_sample = feats[num_sample:num_sample+conf.args.opt['batch_size']], cls[num_sample:num_sample+conf.args.opt['batch_size']], dls[num_sample:num_sample+conf.args.opt['batch_size']]
                self.evaluation_online(num_sample + conf.args.opt['batch_size'], '',
                                       [list(current_sample[0]), list(current_sample[1]), list(current_sample[2])])

        # logging json files
        json_file = open(self.write_path + 'online_eval.json', 'w')
        json_subsample = {key: self.json[key] for key in self.json.keys() - {'extracted_feat'}}
        json_file.write(to_json(json_subsample))
        json_file.close()

    def validation(self, epoch):
        """
        Validate the performance of the model
        """
        class_accuracy_of_test_data, loss, _ = self.evaluation(epoch, 'valid')

        return class_accuracy_of_test_data, loss

    def test(self, epoch):
        """
        Test the performance of the model
        """

        #### for test data
        class_accuracy_of_test_data, loss, cm_class = self.evaluation(epoch, 'test')

        return class_accuracy_of_test_data, loss
