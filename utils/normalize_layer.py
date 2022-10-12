import torch

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2471, 0.2435, 0.2616]

_CIFAR100_MEAN = [0.5071, 0.4865, 0.4409]
_CIFAR100_STDDEV = [0.2673, 0.2564, 0.2762]


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.
      """

    def __init__(self, means, sds):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.register_buffer(
            'mu', torch.tensor(means).view(-1, 1, 1))
        self.register_buffer(
            'sigma', torch.tensor(sds).view(-1, 1, 1))

    def forward(self, input: torch.tensor):
        return (input - self.mu) / self.sigma


def get_normalize_layer(dataset):
    """Return the dataset's normalization layer"""
    if dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == "cifar100":
        return NormalizeLayer(_CIFAR100_MEAN, _CIFAR100_STDDEV)
    else:
        return None