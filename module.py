import torch
from torch import nn
import torch.nn.functional as F
"""
Models Set

1.LSP_1: learnable feature perturbation module
2.CNN: Convolutional Neural Network for Feature Extraction
3.CNN_Tea: Convolutional Neural Network for Feature Extraction with LDP Module
4.Classifier:Classifier for CNNs
5.Classifier_te: Classifier for CNN_Tea
6.SupConLoss: supervised contrast loss
7.CORAL: domain metric

"""
class LSP_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(496, 1)
        self.fc2 = nn.Linear(496, 1)
        self.gamma = nn.Parameter(torch.zeros((40, 16,1)), requires_grad = True)
        self.beta = nn.Parameter(torch.zeros((40, 16,1)), requires_grad = True)

    def forward(self, x):
        h1 = (self.fc1(x))
        h2 = (self.fc2(x))
        gamma = h1.view(h1.size(0), h1.size(1), 1)
        beta = h2.view(h2.size(0), h2.size(1), 1)

        return (1 + gamma) * (x) + beta, gamma, beta

class CNN_Tea(nn.Module):
    def __init__(self):
        super(CNN_Tea, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, 32),
            nn.InstanceNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2))

        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, 3),
            nn.InstanceNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2))

        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, 3),
            nn.InstanceNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4))

        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 128, 3),
            nn.InstanceNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4))

        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, 3),
            nn.InstanceNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2))

        self.lsp_1 = LSP_1()

    def forward(self, x, perturb = False):
        x1 = self.conv1(x)
        if perturb:
            x1, game, beat = self.lsp_1(x1)
        self.l0 = x1
        x2 = self.conv2(x1)
        self.l1 = x2
        x3 = self.conv3(x2)
        self.l2 = x3
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        fea = x5.view(x5.size(0), -1)
        return fea, x3

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, 32),
            nn.InstanceNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2))

        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, 3),
            nn.InstanceNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2))

        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, 3),
            nn.InstanceNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4))

        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 128, 3),
            nn.InstanceNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4))

        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, 3),
            nn.InstanceNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2))

    def forward(self, x, train=True):
        x1 = self.conv1(x)
        self.l0 = x1
        x2 = self.conv2(x1)
        self.l1 = x2
        x3 = self.conv3(x2)
        self.l2 = x3
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        fea = x5.view(x5.size(0), -1)

        return fea, x3

class Classifier(nn.Module):
    def __init__(self,n_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(nn.Linear(128*6, 128))
        self.out = nn.Linear(128, n_classes)

    def forward(self, x):
        fea = self.fc(x)
        label = self.out(fea)
        return label

class Classifier_te(nn.Module):
    def __init__(self,n_classes):
        super(Classifier_te, self).__init__()
        self.fc = nn.Sequential(nn.Linear(128*6, 128))
        self.out = nn.Linear(128, n_classes)

    def forward(self, x):
        fea= self.fc(x)
        label = self.out(fea)
        return label

class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class InstanceWhitening(nn.Module):

    def __init__(self):
        super(InstanceWhitening, self).__init__()
        self.instance_standardization = nn.InstanceNorm1d(64, affine=False)

    def forward(self, x):
        x = self.instance_standardization(x)
        return x

def get_covariance_matrix(f_map, eye=None):
    eps = 1e-5
    B, C, H = f_map.shape
    HW = H
    if eye is None:
        eye = torch.eye(C).cuda()
    f_map = f_map.contiguous().view(B, C, -1)
    f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW - 1) + (eps * eye)
    return f_cor, B

def get_cross_covariance_matrix(f_map1, f_map2, eye=None):
    eps = 1e-5
    assert f_map1.shape == f_map2.shape
    B, C, H= f_map1.shape
    HW = H
    if eye is None:
        eye = torch.eye(C).cuda()
    f_map1 = f_map1.contiguous().view(B, C, -1)
    f_map2 = f_map2.contiguous().view(B, C, -1)
    f_cor = torch.bmm(f_map1, f_map2.transpose(1, 2)).div(HW - 1) + (eps * eye)
    return f_cor, B

def cross_whitening_loss(k_feat, q_feat):
    assert k_feat.shape == q_feat.shape

    f_cor, B = get_cross_covariance_matrix(k_feat, q_feat)
    diag_loss = torch.FloatTensor([0]).cuda()

    for cor in f_cor:
        diag = torch.diagonal(cor.squeeze(dim=0), 0)
        eye = torch.ones_like(diag).cuda()
        diag_loss = diag_loss + F.mse_loss(diag, eye)
    diag_loss = diag_loss / B

    return diag_loss

def CORAL(source, target):

    d = source.shape[1]
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss/(4*d*d)
    return loss


def calculate_recall(y_true, y_pred, num_classes):

    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    recall_per_class = []
    for cls in range(num_classes):

        TP = ((y_pred == cls) & (y_true == cls)).sum().item()
        FN = ((y_pred != cls) & (y_true == cls)).sum().item()
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        recall_per_class.append(recall)
    recall_per_class = torch.tensor(recall_per_class)
    return recall_per_class