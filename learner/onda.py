import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import conf
from .dnn import DNN
from torch.utils.data import DataLoader

from utils.loss_functions import *

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
from utils.iabn import *

class ONDA(DNN):

    def __init__(self, *args, **kwargs):
        super(ONDA, self).__init__(*args, **kwargs)

        for param in self.net.parameters():  # initially turn off requires_grad for all
            param.requires_grad = False
        for module in self.net.modules():

            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

                assert conf.args.use_learned_stats,"ONDA must use momentum-based update"
                if conf.args.use_learned_stats:
                    module.track_running_stats = True
                    module.momentum = conf.args.bn_momentum



    def train_online(self, current_num_sample):
        """
        Train the model
        """

        TRAINED = 0
        SKIPPED = 1
        FINISHED = 2

        if not hasattr(self, 'previous_train_loss'):
            self.previous_train_loss = 0

        if current_num_sample > len(self.target_train_set[0]):
            return FINISHED

        # Add a sample
        feats, cls, dls = self.target_train_set
        current_sample = feats[current_num_sample - 1], cls[current_num_sample - 1], dls[current_num_sample - 1]
        self.mem.add_instance(current_sample)

        if conf.args.use_learned_stats: #batch-free inference
            self.evaluation_online(current_num_sample, '', [[current_sample[0]], [current_sample[1]], [current_sample[2]]])

        if current_num_sample % conf.args.update_every_x != 0:  # train only when enough samples are collected
            if not (current_num_sample == len(self.target_train_set[
                                                  0]) and conf.args.update_every_x >= current_num_sample):  # update with entire data

                self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=self.previous_train_loss)
                return SKIPPED

        # setup models

        self.net.train()

        if len(feats) == 1:  # avoid BN error
            self.net.eval()

        feats, cls, dls = self.mem.get_memory()
        feats, cls, dls = torch.stack(feats), cls, torch.stack(dls)
        dataset = torch.utils.data.TensorDataset(feats)
        data_loader = DataLoader(dataset, batch_size=conf.args.opt['batch_size'],
                                 shuffle=True, drop_last=False, pin_memory=False)

        for e in range(conf.args.epoch):

            for batch_idx, (feats,) in enumerate(data_loader):
                feats = feats.to(device)
                _ = self.net(feats) # update bn stats

        self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=0)


        return TRAINED
