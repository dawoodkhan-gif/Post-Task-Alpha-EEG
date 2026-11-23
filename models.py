"""Compact EEG networks used in this study (PyTorch).

EEGNet: Lawhern et al. (2018), J. Neural Eng. 15, 056013.
ShallowConvNet: Schirrmeister et al. (2017), Hum. Brain Mapp. 38, 5391-5420.
Kernel and pooling lengths are scaled to the 128 Hz sampling rate used here.
Each model returns (logit, embedding), the embedding being the input to the final linear layer.
"""
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, n_channels=64, n_times=256, F1=8, D=2, F2=16, kernel=64, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel), padding=(0, kernel // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_feat = self.features(torch.zeros(1, 1, n_channels, n_times)).shape[1]
        self.classifier = nn.Linear(n_feat, 1)

    def forward(self, x):
        z = self.features(x.unsqueeze(1))
        return self.classifier(z).squeeze(1), z


class Square(nn.Module):
    def forward(self, x):
        return x * x


class SafeLog(nn.Module):
    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-6))


class ShallowConvNet(nn.Module):
    def __init__(self, n_channels=64, n_times=256, n_filters=40, kernel=13, pool=38, stride=8, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, n_filters, (1, kernel)),
            nn.Conv2d(n_filters, n_filters, (n_channels, 1), bias=False),
            nn.BatchNorm2d(n_filters),
            Square(),
            nn.AvgPool2d((1, pool), stride=(1, stride)),
            SafeLog(),
            nn.Dropout(dropout),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_feat = self.features(torch.zeros(1, 1, n_channels, n_times)).shape[1]
        self.classifier = nn.Linear(n_feat, 1)

    def forward(self, x):
        z = self.features(x.unsqueeze(1))
        return self.classifier(z).squeeze(1), z


MODELS = {'EEGNet': EEGNet, 'ShallowConvNet': ShallowConvNet}
