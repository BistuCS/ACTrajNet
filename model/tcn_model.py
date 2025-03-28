
"""Modified from https://github.com/locuslab/TCN"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import argparse

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.1)
        self.conv2.weight.data.normal_(0, 0.1)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.tanh(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        # num_channels = [256, 256]
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
            # print(out_channels)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


if __name__=='__main__':
    print(__name__)
    parser=argparse.ArgumentParser()
    parser.add_argument('--input_channels',type=int,default=3)
    parser.add_argument('--tcn_channel_size',type=int,default=256)
    args=parser.parse_args()
    input_size = args.input_channels
    n_classes = 12
    num_channels= [args.tcn_channel_size]*2  #[256] * 2
    num_channels.append(n_classes)
    tcn_kernel_size = 4
    dropout = 0.6
    x = torch.randn(1, 3, 11)
    tcn_encoder_x = TemporalConvNet(input_size, num_channels, kernel_size=tcn_kernel_size, dropout=dropout)
    # print(x)
    print(x.shape)
    y = tcn_encoder_x(x)
    print('tcn shape', y.shape) # (1, 12, 11)
    fc_a_X = nn.Linear(11, 12) 
    a = torch.Tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    s = fc_a_X(a)
    s = s[None, :, None]
    print(a.shape)
    sig_a_X = nn.Sigmoid()

    print(s.shape)
    s = sig_a_X(s)
    print('after sig', s.shape)
    print(y.shape, s.shape)
    # print(y)
    # s = torch.transpose(s, 1, 2)
    print(s * y)
    encoded_y = torch.flatten(y)[None,None,:] # (1, 1, 132)
    print('encoded_y shape', encoded_y.shape)
    