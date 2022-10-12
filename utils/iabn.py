import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import conf
import copy

def convert_iabn(module, **kwargs):
    module_output = module
    if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        IABN = InstanceAwareBatchNorm2d if isinstance(module, nn.BatchNorm2d) else InstanceAwareBatchNorm1d
        module_output = IABN(
            num_channels=module.num_features,
            k=conf.args.iabn_k,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
        )

        module_output._bn = copy.deepcopy(module)

    for name, child in module.named_children():
        module_output.add_module(
            name, convert_iabn(child, **kwargs)
        )
    del module
    return module_output


class InstanceAwareBatchNorm2d(nn.Module):
    def __init__(self, num_channels, k=3.0, eps=1e-5, momentum=0.1, affine=True):
        super(InstanceAwareBatchNorm2d, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.k=k
        self.affine = affine
        self._bn = nn.BatchNorm2d(num_channels, eps=eps,
                                  momentum=momentum, affine=affine)

    def _softshrink(self, x, lbd):
        x_p = F.relu(x - lbd, inplace=True)
        x_n = F.relu(-(x + lbd), inplace=True)
        y = x_p - x_n
        return y

    def forward(self, x):
        b, c, h, w = x.size()
        sigma2, mu = torch.var_mean(x, dim=[2, 3], keepdim=True, unbiased=True) #IN

        if self.training:
            _ = self._bn(x)
            sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2, 3], keepdim=True, unbiased=True)
        else:
            if self._bn.track_running_stats == False and self._bn.running_mean is None and self._bn.running_var is None: # use batch stats
                sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2, 3], keepdim=True, unbiased=True)
            else:
                mu_b = self._bn.running_mean.view(1, c, 1, 1)
                sigma2_b = self._bn.running_var.view(1, c, 1, 1)


        if h*w <=conf.args.skip_thres:
            mu_adj = mu_b
            sigma2_adj = sigma2_b
        else:
            s_mu = torch.sqrt((sigma2_b + self.eps) / (h * w))
            s_sigma2 = (sigma2_b + self.eps) * np.sqrt(2 / (h * w - 1))

            mu_adj = mu_b + self._softshrink(mu - mu_b, self.k * s_mu)

            sigma2_adj = sigma2_b + self._softshrink(sigma2 - sigma2_b, self.k * s_sigma2)

            sigma2_adj = F.relu(sigma2_adj) #non negative

        x_n = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)
        if self.affine:
            weight = self._bn.weight.view(c, 1, 1)
            bias = self._bn.bias.view(c, 1, 1)
            x_n = x_n * weight + bias
        return x_n


class InstanceAwareBatchNorm1d(nn.Module):
    def __init__(self, num_channels, k=3.0, eps=1e-5, momentum=0.1, affine=True):
        super(InstanceAwareBatchNorm1d, self).__init__()
        self.num_channels = num_channels
        self.k = k
        self.eps = eps
        self.affine = affine
        self._bn = nn.BatchNorm1d(num_channels, eps=eps,
                                  momentum=momentum, affine=affine)

    def _softshrink(self, x, lbd):
        x_p = F.relu(x - lbd, inplace=True)
        x_n = F.relu(-(x + lbd), inplace=True)
        y = x_p - x_n
        return y

    def forward(self, x):
        b, c, l = x.size()
        sigma2, mu = torch.var_mean(x, dim=[2], keepdim=True, unbiased=True)
        if self.training:
            _ = self._bn(x)
            sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2], keepdim=True, unbiased=True)
        else:
            if self._bn.track_running_stats == False and self._bn.running_mean is None and self._bn.running_var is None: # use batch stats
                sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2], keepdim=True, unbiased=True)
            else:
                mu_b = self._bn.running_mean.view(1, c, 1)
                sigma2_b = self._bn.running_var.view(1, c, 1)

        if l <=conf.args.skip_thres:
            mu_adj = mu_b
            sigma2_adj = sigma2_b
        
        else:
            s_mu = torch.sqrt((sigma2_b + self.eps) / l) ##
            s_sigma2 = (sigma2_b + self.eps) * np.sqrt(2 / (l - 1))

            mu_adj = mu_b + self._softshrink(mu - mu_b, self.k * s_mu)
            sigma2_adj = sigma2_b + self._softshrink(sigma2 - sigma2_b, self.k * s_sigma2)
            sigma2_adj = F.relu(sigma2_adj)


        x_n = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)

        if self.affine:
            weight = self._bn.weight.view(c, 1)
            bias = self._bn.bias.view(c, 1)
            x_n = x_n * weight + bias

        return x_n
