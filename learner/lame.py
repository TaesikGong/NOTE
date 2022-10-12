import conf
from .dnn import DNN
from torch.utils.data import DataLoader
from utils.normalize_layer import *
from utils.loss_functions import *

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")


class AffinityMatrix:

    def __init__(self, **kwargs):
        pass

    def __call__(X, **kwargs):
        raise NotImplementedError

    def is_psd(self, mat):
        eigenvalues = torch.eig(mat)[0][:, 0].sort(descending=True)[0]
        return eigenvalues, float((mat == mat.t()).all() and (eigenvalues >= 0).all())

    def symmetrize(self, mat):
        return 1 / 2 * (mat + mat.t())


class kNN_affinity(AffinityMatrix):
    def __init__(self, knn: int, **kwargs):
        self.knn = knn

    def __call__(self, X):
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
        n_neighbors = min(self.knn + 1, N)

        knn_index = dist.topk(n_neighbors, -1, largest=False).indices[:, 1:]  # [N, knn]

        W = torch.zeros(N, N, device=X.device)
        W.scatter_(dim=-1, index=knn_index, value=1.0)

        return W


class rbf_affinity(AffinityMatrix):
    def __init__(self, sigma: float, **kwargs):
        self.sigma = sigma
        self.k = kwargs['knn']

    def __call__(self, X):
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
        n_neighbors = min(self.k, N)
        kth_dist = dist.topk(k=n_neighbors, dim=-1, largest=False).values[:,
                   -1]  # compute k^th distance for each point, [N, knn + 1]
        sigma = kth_dist.mean()
        rbf = torch.exp(- dist ** 2 / (2 * sigma ** 2))
        # mask = torch.eye(X.size(0)).to(X.device)
        # rbf = rbf * (1 - mask)
        return rbf


class linear_affinity(AffinityMatrix):

    def __call__(self, X: torch.Tensor):
        """
        X: [N, d]
        """
        return torch.matmul(X, X.t())


class LAME(DNN):
    def __init__(self, *args, **kwargs):
        super(LAME, self).__init__(*args, **kwargs)

        self.knn = 5
        self.sigma = 1.0  # from overall_best.yaml in LAME github
        self.affinity = kNN_affinity(knn=self.knn)
        if isinstance(self.net, nn.Sequential):
            if isinstance(self.net[0], NormalizeLayer):
                self.featurizer = torch.nn.Sequential(self.net[0], *list(self.net[1].children())[:-1])
        else:
            self.featurizer = torch.nn.Sequential(*list(self.net.children())[:-1])

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

        # Get sample from target
        xs, cls, dls = self.target_train_set
        current_sample = xs[current_num_sample - 1], cls[current_num_sample - 1], dls[current_num_sample - 1]

        # Add sample to memory
        self.mem.add_instance(current_sample)

        # Skipping depend on "batch size"
        if current_num_sample % conf.args.update_every_x != 0:  # train only when enough samples are collected
            if not (current_num_sample == len(self.target_train_set[
                                                  0]) and conf.args.update_every_x >= current_num_sample):  # update with entire data

                self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=self.previous_train_loss)
                return SKIPPED

        # setup models
        self.net.eval()

        if len(xs) == 1:  # avoid BN error
            self.net.eval()

        xs, cls, dls = self.mem.get_memory()
        xs, cls, dls = torch.stack(xs), torch.stack(cls), torch.stack(dls)

        dataset = torch.utils.data.TensorDataset(xs, cls, dls)
        data_loader = DataLoader(dataset, batch_size=conf.args.opt['batch_size'],
                                 shuffle=True,
                                 drop_last=False, pin_memory=False)

        self.evaluation_online(current_num_sample, '', self.mem.get_memory())

        self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=0)

        return TRAINED


    def batch_evaluation(self, extracted_feat):
        out = self.net(extracted_feat)
        unary = -torch.log(out.softmax(-1) + 1e-10)  # softmax the output

        if conf.args.model == 'resnet18':
            feats = self.featurizer(extracted_feat)
            feats = F.avg_pool2d(feats, 4)
            feats = feats.view(feats.size(0), -1)
        else:
            feats = torch.nn.functional.normalize(self.featurizer(extracted_feat), p=2, dim=-1).squeeze()

        kernel = self.affinity(feats)

        Y = laplacian_optimization(unary, kernel)
        return Y


def laplacian_optimization(unary, kernel, bound_lambda=1, max_steps=100):
    E_list = []
    oldE = float('inf')
    Y = (-unary).softmax(-1)  # [N, K]
    for i in range(max_steps):
        pairwise = bound_lambda * kernel.matmul(Y)  # [N, K]
        exponent = -unary + pairwise
        Y = exponent.softmax(-1)
        E = entropy_energy(Y, unary, pairwise, bound_lambda).item()
        E_list.append(E)

        if (i > 1 and (abs(E - oldE) <= 1e-8 * abs(oldE))):
            break
        else:
            oldE = E

    return Y


def entropy_energy(Y, unary, pairwise, bound_lambda):
    E = (unary * Y - bound_lambda * pairwise * Y + Y * torch.log(Y.clip(1e-20))).sum()
    return E
