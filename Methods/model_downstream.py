from torch import nn
from Methods.tcn import TemporalConvNet
import numpy as np
import torch
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels,kernel_size,mlp_hid_size, dropout,if_fix=False):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels,kernel_size=kernel_size, dropout=dropout)
        if if_fix:
            for l in range(0,len(num_channels)):
                for p in self.tcn.network[l].parameters():
                    p.requires_grad = False
        self.linear_a = nn.Linear(num_channels[-1], mlp_hid_size)
        self.linear_b = nn.Linear(mlp_hid_size, int(mlp_hid_size*1.0/2))
        self.linear_c = nn.Linear(int(mlp_hid_size*1.0/2), 1)
        self.leakyrelu = nn.LeakyReLU()

    def loss(self,true_y,pred_y):
        self.loss_value = torch.mean(torch.pow((true_y-pred_y)/true_y,2))
        return self.loss_value

    def forward(self, x):
        y1 = self.tcn(x)
        y2 = self.leakyrelu(self.linear_c(self.leakyrelu(self.linear_b(self.leakyrelu(self.linear_a(y1[:, :, -1]))))))
        return y2
