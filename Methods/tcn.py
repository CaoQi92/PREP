import torch
import torch.nn as nn
from torch.nn.utils import weight_norm,remove_weight_norm
import numpy as np

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self,n_inputs, n_outputs, kernel_size,stride, dilation, padding,dropout=0.2):
        super(TemporalBlock, self).__init__()
        # self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
        #                                    stride=stride, padding=padding, dilation=dilation))

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        # self.eta1 = torch.zeros(self.conv1.weight.size())+1e-20
        # self.eta1 = self.eta1.to(device)
        # self.conv1.weight.data = self.conv1.weight.data + eta1
        # self.wn1 = weight_norm(self.conv1)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
        #                                    stride=stride, padding=padding, dilation=dilation))
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        # self.eta2 = torch.zeros(self.conv2.weight.size())+1e-20
        # self.conv2.weight.data = self.conv2.weight.data + eta2
        # self.wn2 = weight_norm(self.conv2)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)



        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
        self.dilation = dilation
        # self.device = device

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        # remove_weight_norm(self.wn1)
        # remove_weight_norm(self.wn2)
        # if self.conv1.weight.data.is_cuda:
        #     self.eta1 = self.eta1.to(self.device)
        #     self.eta2 = self.eta2.to(self.device)
        # # self.conv1.weight.data = self.conv1.weight.data + self.eta1
        # # self.conv2.weight.data = self.conv2.weight.data +self.eta2
        # print("conv1:",self.conv1.weight.data.is_cuda)
        # print("eta1:",self.eta1.is_cuda)
        # self.conv1.weight.data = torch.max(self.conv1.weight.data, self.eta1)
        # self.wn1 = weight_norm(self.conv1)
        # print("conv1:", self.conv1.weight.data.is_cuda)
        #
        # self.conv2.weight.data = torch.max(self.conv2.weight.data, self.eta2)
        # self.wn2 = weight_norm(self.conv2)


        # con1 = self.conv1(x)
        # # con1 = self.wn1(x)
        # cho1 = self.chomp1(con1)
        # r1 = self.relu1(cho1)
        # d1 = self.dropout1(r1)
        # con2 = self.conv2(d1)
        # # con2 = self.wn2(d1)
        # cho2 = self.chomp2(con2)
        # r2 = self.relu2(cho2)
        # out = self.dropout2(r2)
        res = x if self.downsample is None else self.downsample(x)

        # if np.sum(np.isnan(con1.detach().cpu().numpy())) > 0:
        #     print(self.dilation, "model inputs con1 nan:", con1.size(), "input:", con1)
        # if np.sum(np.isnan(cho1.detach().cpu().numpy())) > 0:
        #     print(self.dilation,"model inputs cho1 nan:", cho1.size(), "input:", cho1)
        # if np.sum(np.isnan(r1.detach().cpu().numpy())) > 0:
        #     print(self.dilation,"model inputs r1 nan:", r1.size(), "input:", r1)
        # if np.sum(np.isnan(d1.detach().cpu().numpy())) > 0:
        #     print(self.dilation,"model inputs d1 nan:", d1.size(), "input:", d1)
        # if np.sum(np.isnan(con2.detach().cpu().numpy())) > 0:
        #     print(self.dilation,"model inputs con2 nan:", con2.size(), "input:", con2)
        # if np.sum(np.isnan(cho2.detach().cpu().numpy())) > 0:
        #     print(self.dilation,"model inputs cho2 nan:", cho2.size(), "input:", cho2)
        # if np.sum(np.isnan(r2.detach().cpu().numpy())) > 0:
        #     print(self.dilation,"model inputs r2 nan:", r2.size(), "input:", r2)
        # if np.sum(np.isnan(out.detach().cpu().numpy())) > 0:
        #     print(self.dilation,"model inputs out nan:", out.size(), "input:", out)
        # if np.sum(np.isnan(res.detach().cpu().numpy())) > 0:
        #     print(self.dilation,"model inputs res nan:", res.size(), "input:", res)

        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self,num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
