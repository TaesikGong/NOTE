import torch
import torch.nn as nn
import numpy as np
import math
import conf
from torch.utils.data import DataLoader
from .dnn import DNN

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


class BN_Stats(DNN):

    def __init__(self, *args, **kwargs):
        super(BN_Stats, self).__init__(*args, **kwargs)

        # turn on grad for BN params only
        for module in self.net.modules():
                # print(module)
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

                if conf.args.use_learned_stats: # use learned stats + update with target via momentum
                    module.track_running_stats = True
                    module.momentum = conf.args.bn_momentum

                else: #Default: use the target batch stats
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None

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

        assert (conf.args.update_every_x == conf.args.memory_size)

        if current_num_sample % conf.args.update_every_x != 0:  # train only when enough samples are collected
            if not (current_num_sample == len(self.target_train_set[
                                                  0]) and conf.args.update_every_x >= current_num_sample):  # update with entire data
                self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=self.previous_train_loss)
                return SKIPPED


        # Evaluate with a batch
        self.evaluation_online(current_num_sample, '', self.mem.get_memory())

        self.batch_instances = []

        if conf.args.use_learned_stats: # update batch stats with momentum
            self.net.train()

            feats, cls, dls = self.mem.get_memory()
            feats, cls, dls = torch.stack(feats), cls, torch.stack(dls)
            dataset = torch.utils.data.TensorDataset(feats)
            data_loader = DataLoader(dataset, batch_size=conf.args.opt['batch_size'],
                                     shuffle=True,
                                     drop_last=False, pin_memory=False)

            with torch.no_grad():

                for batch_idx, (feats,) in enumerate(data_loader):
                    feats = feats.to(device)
                    _ = self.net(feats) # update batch stats

        ################################ TT_Single_stats: Mazankiewicz, A., Böhm, K., & Bergés, M. (2020). Incremental Real-Time Personalization In Human Activity Recognition Using Domain Adaptive Batch Normalization. ArXiv, 4(4). https://doi.org/10.1145/3432230

        self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=0)

        return TRAINED
