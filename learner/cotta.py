import copy

import conf
from .dnn import DNN
from torch.utils.data import DataLoader

from utils.loss_functions import *
import PIL
from utils import cotta_utils
import torchvision.transforms as transforms

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")

@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    if conf.args.dataset in ['cifar10', 'cifar100']:
        img_shape = (32, 32, 3)
    else:
        img_shape = (224, 224, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        cotta_utils.Clip(0.0, 1.0),
        cotta_utils.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        cotta_utils.GaussianNoise(0, gaussian_std),
        cotta_utils.Clip(clip_min, clip_max)
    ])
    return tta_transforms


class CoTTA(DNN):
    def __init__(self, *args, **kwargs):
        super(CoTTA, self).__init__(*args, **kwargs)

        for param in self.net.parameters():  #turn on requires_grad for all
            param.requires_grad = True

        for module in self.net.modules():

            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                #use of batch stats in train and eval modes: https://github.com/qinenergy/cotta/blob/main/cifar/cotta.py
                if conf.args.use_learned_stats:
                    module.track_running_stats = True
                    module.momentum = conf.args.bn_momentum
                else:
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

        self.mt = conf.args.ema_factor #0.999 for every dataset
        self.rst = conf.args.restoration_factor #0.01 for all dataset
        self.ap = conf.args.aug_threshold #0.92 for CIFAR10, 0.72 for CIFAR100
        self.episodic = False

        self.net_state = copy.deepcopy(self.net.state_dict())
        self.net_anchor = copy.deepcopy(self.net)
        self.net_ema = copy.deepcopy(self.net)
        self.transform = get_tta_transforms()

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

        # Evaluate with a batch
        if not conf.args.use_learned_stats: #batch-based inference
            self.evaluation_online(current_num_sample, '', self.mem.get_memory())


        # setup models
        self.net.train()

        if len(feats) == 1:  # avoid BN error
            self.net.eval()

        feats, cls, dls = self.mem.get_memory()
        feats, cls, dls = torch.stack(feats), cls, torch.stack(dls)



        dataset = torch.utils.data.TensorDataset(feats)
        data_loader = DataLoader(dataset, batch_size=conf.args.opt['batch_size'],
                                 shuffle=True,
                                 drop_last=False, pin_memory=False)

        for e in range(conf.args.epoch):

            for batch_idx, (x,) in enumerate(data_loader):
                x = x.to(device)

                outputs = self.net(x)

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
                # Student update
                loss = (softmax_entropy(outputs, outputs_ema)).mean(0)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                # Teacher update
                self.net_ema = update_ema_variables(ema_model=self.net_ema, model=self.net, alpha_teacher=self.mt)
                # Stochastic restore
                if True:
                    for nm, m in self.net.named_modules():
                        for npp, p in m.named_parameters():
                            if npp in ['weight', 'bias'] and p.requires_grad:
                                mask = (torch.rand(p.shape) < self.rst).float().cuda()
                                with torch.no_grad():
                                    p.data = self.net_state[f"{nm}.{npp}"] * mask + p * (1. - mask)

        self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=0)

        return TRAINED