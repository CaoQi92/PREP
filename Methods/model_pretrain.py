from torch import nn
from Methods.tcn import TemporalConvNet
import numpy as np
import torch

class TimeSlicePre(nn.Module):
    def __init__(self, input_size, output_size, num_channels,kernel_size, mlp_hid_size,dropout):
        super(TimeSlicePre, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels,kernel_size=kernel_size, dropout=dropout)
        self.linear1 = nn.Linear(2*num_channels[-1], mlp_hid_size)
        self.linear2 = nn.Linear(mlp_hid_size,int(mlp_hid_size/2))
        self.linear3 = nn.Linear(int(mlp_hid_size/2),1)
        self.leakyrelu = nn.LeakyReLU()
    def get_concat(self):
        return self.y_concat
    def forward(self, x1, x2):
        # batch_size * num_channels * seq_length
        y1_out = self.tcn(x1)
        # batch_size * num_channels
        y1_out_last = y1_out[:,:,-1]

        y2_out = self.tcn(x2)
        y2_out_last = y2_out[:,:,-1] 
        
        self.y_concat = torch.cat((y1_out_last,y2_out_last),1)
        result = self.leakyrelu(self.linear3(self.leakyrelu(self.linear2(self.leakyrelu(self.linear1(self.y_concat))))))

        return result
