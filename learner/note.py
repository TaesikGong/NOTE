
import conf
from .dnn import DNN
from torch.utils.data import DataLoader

from utils import memory

from utils.loss_functions import *

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
from utils.iabn import *

class NOTE(DNN):

    def __init__(self, *args, **kwargs):
        super(NOTE, self).__init__(*args, **kwargs)

        # turn on grad for BN params only

        for param in self.net.parameters():  # initially turn off requires_grad for all
                param.requires_grad = False
        for module in self.net.modules():

            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

                if conf.args.use_learned_stats:
                    module.track_running_stats = True
                    module.momentum = conf.args.bn_momentum
                else:
                    # With below, this module always uses the test batch statistics (no momentum)
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            elif isinstance(module, nn.InstanceNorm1d) or isinstance(module, nn.InstanceNorm2d): #ablation study
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            if conf.args.iabn:
                if isinstance(module, InstanceAwareBatchNorm2d) or isinstance(module, InstanceAwareBatchNorm1d):
                    for param in module.parameters():
                        param.requires_grad = True


        self.fifo = memory.FIFO(capacity=conf.args.update_every_x) # required for evaluation


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
        self.fifo.add_instance(current_sample)#for batch-based inferece

        with torch.no_grad():

            self.net.eval()

            if conf.args.memory_type in ['FIFO', 'Reservoir']:
                self.mem.add_instance(current_sample)

            elif conf.args.memory_type in ['PBRS']:
                f, c, d = current_sample[0].to(device), current_sample[1].to(device), current_sample[2].to(device)

                logit = self.net(f.unsqueeze(0))
                pseudo_cls = logit.max(1, keepdim=False)[1][0]
                self.mem.add_instance([f, pseudo_cls, d, c, 0])

        if conf.args.use_learned_stats: #batch-free inference
            self.evaluation_online(current_num_sample, '', [[current_sample[0]], [current_sample[1]], [current_sample[2]]])

        if current_num_sample % conf.args.update_every_x != 0:  # train only when enough samples are collected
            if not (current_num_sample == len(self.target_train_set[
                                                  0]) and conf.args.update_every_x >= current_num_sample):  # update with entire data

                self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=self.previous_train_loss)
                return SKIPPED

        if not conf.args.use_learned_stats: #batch-based inference
            self.evaluation_online(current_num_sample, '', self.fifo.get_memory())


        if conf.args.no_adapt: # for ablation
            return TRAINED
        # setup models
        self.net.train()

        if len(feats) == 1:  # avoid BN error
            self.net.eval()

        feats, _, _ = self.mem.get_memory()
        feats = torch.stack(feats)
        dataset = torch.utils.data.TensorDataset(feats)
        data_loader = DataLoader(dataset, batch_size=conf.args.opt['batch_size'],
                                 shuffle=True, drop_last=False, pin_memory=False)

        entropy_loss = HLoss(temp_factor=conf.args.temperature)

        for e in range(conf.args.epoch):

            for batch_idx, (feats,) in enumerate(data_loader):
                feats = feats.to(device)
                preds_of_data = self.net(feats) # update bn stats


                if conf.args.no_optim:
                    pass # no optimization
                else:
                    loss = entropy_loss(preds_of_data)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=0)

        return TRAINED
