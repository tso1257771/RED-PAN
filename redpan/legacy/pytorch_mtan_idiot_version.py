import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels,
            kernel_size, padding=1, stride=None):
        super(conv_block, self).__init__()

        self.conv1d = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            padding=padding, stride=stride)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        out = self.conv1d(x)
        out = self.batchnorm(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class RRconv_block(nn.Module):
    def __init__(self, in_channels, out_channels, 
            kernel_size, padding=1, stride=None, 
            RRconv_time=3):
        super(RRconv_block, self).__init__()

        self.RRconv_time = RRconv_time
        self.conv_block_init = conv_block(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size,
                padding=padding, stride=stride)

        self.conv_1x1 = nn.Conv1d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=1, padding=0, stride=1)
        
        self.conv_block_res = conv_block(
                in_channels=out_channels, 
                out_channels=out_channels, 
                kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        conv_out = self.conv_block_init(x)
        conv_out_1x1 = self.conv_1x1(conv_out)

        for i in range(self.RRconv_time):
            if i == 0:
                res_unit = conv_out
            res_unit += conv_out
            res_unit = self.conv_block_res(res_unit)
        RRconv_out = res_unit + conv_out_1x1
        return RRconv_out

class upconv_concat_RRblock(nn.Module):
    def __init__(self, in_channels, out_channels, upsize=5, RRconv_time=3):
        super(upconv_concat_RRblock, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=upsize)
        self.conv_block = conv_block(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=1, padding=1, stride=1)
        self.conv_skip_conncet = nn.Conv1d(
            in_channels=out_channels*2, 
            out_channels=out_channels, 
            kernel_size=1, padding=0, stride=1)
        self.RRconv_block = RRconv_block(in_channels=out_channels,
            out_channels=out_channels, kernel_size=1, padding=0,
            stride=1, RRconv_time=RRconv_time)
    
    def forward(self, target_layer=None, concat_layer=None):
        up_conv_out = self.upsample(target_layer)
        up_conv_out = self.conv_block(up_conv_out)
        skip_connect = torch.cat([
            up_conv_out[:, :, :concat_layer.shape[2]], 
            concat_layer], dim=1)
        conca_out = self.conv_skip_conncet(skip_connect)
        RR_out = self.RRconv_block(conca_out)
        return RR_out

class att_layer(nn.Module):
    def __init__(self, channel):
        super(att_layer, self).__init__()

        self.batchnorm = nn.BatchNorm1d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv_1x1 = nn.Conv1d(in_channels=channel,
            out_channels=channel, kernel_size=1, padding=0,
            stride=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.batchnorm(x)
        out = self.relu(out)
        out = self.conv_1x1(out)
        out = self.batchnorm(out)
        out = self.sigmoid(out)
        return out

class mtan_att_block(nn.Module):
    def __init__(self, in_channels, out_channels,
            upsize=5, stride=5, mode=None):
        super(mtan_att_block, self).__init__()
        
        # channels number of previous att/backbone layer
        self.in_chanels = in_channels 
        # channels number of current backbone layer 
        self.out_channels = out_channels
        self.mode = mode

        if mode == 'up':
            self.init_conv = nn.Sequential(
                nn.Upsample(scale_factor=upsize),
                conv_block(
                    in_channels=in_channels*2, # due to concatenation
                    out_channels=out_channels,
                    kernel_size=1, padding=1, stride=1
                )
            )
        if mode == 'down':
            self.init_conv = nn.Conv1d(
                in_channels=in_channels*2, # due to concatenation
                out_channels=out_channels, 
                kernel_size=1, padding=0, stride=stride)

        self.att_layer = att_layer(channel=self.out_channels)

    def forward(self, pre_att_layer=None, pre_target_layer=None, 
            target_layer=None):

        pre_layer_concat = torch.cat(
                [pre_att_layer, pre_target_layer], dim=1)
        # attention layer
        pre_conv_init = self.init_conv(pre_layer_concat)
        att = self.att_layer(pre_conv_init)

        ## element-wise multiplication with target layer
        att_gate = torch.multiply(
            att[:, :, :target_layer.shape[2]], target_layer)

        return att_gate

n_chn = [3, 6, 12, 18, 24, 30, 36]
x = torch.randn(50, 3, 6000) # batch_size, channel, data_length

x1_1 = conv_block(in_channels=3, out_channels=6, 
    kernel_size=1, padding=0, stride=1)(x)
x1_2 = RRconv_block(in_channels=6, out_channels=6, 
    kernel_size=1, padding=0, stride=1, RRconv_time=3)(x1_1)
# (Batch, 6, 6000)

x2_1 = conv_block(in_channels=6, out_channels=12, 
    kernel_size=5, padding=1, stride=5)(x1_2)
x2_2 = RRconv_block(in_channels=12, out_channels=12, 
    kernel_size=1, padding=0, stride=1, RRconv_time=3)(x2_1)
# (Batch, 12, 1200)

x3_1 = conv_block(in_channels=12, out_channels=18, 
    kernel_size=5, padding=1, stride=5)(x2_2)
x3_2 = RRconv_block(in_channels=18, out_channels=18, 
    kernel_size=1, padding=0, stride=1, RRconv_time=3)(x3_1)
# (Batch, 18, 240)

x4_1 = conv_block(in_channels=18, out_channels=24, 
    kernel_size=5, padding=1, stride=5)(x3_2)
x4_2 = RRconv_block(in_channels=24, out_channels=24, 
    kernel_size=1, padding=0, stride=1, RRconv_time=3)(x4_1)
# (Batch, 24, 48)

x5_1 = conv_block(in_channels=24, out_channels=30, 
    kernel_size=5, padding=1, stride=5)(x4_2)
x5_2 = RRconv_block(in_channels=30, out_channels=30, 
    kernel_size=1, padding=0, stride=1, RRconv_time=3)(x5_1)
#(Batch, 30, 10)

x_bottleneck_1 = conv_block(in_channels=30, out_channels=36, 
    kernel_size=5, padding=1, stride=5)(x5_2)
x_bottleneck_2 = RRconv_block(in_channels=36, out_channels=36, 
    kernel_size=1, padding=0, stride=1, RRconv_time=3)(x_bottleneck_1)
# (Batch, 36, 2)

u1 = upconv_concat_RRblock(in_channels=36, out_channels=30, 
    upsize=5)(target_layer=x_bottleneck_2, concat_layer=x5_2)

u2 = upconv_concat_RRblock(in_channels=30, out_channels=24, 
    upsize=5)(target_layer=u1, concat_layer=x4_2)

u3 = upconv_concat_RRblock(in_channels=24, out_channels=18, 
    upsize=5)(target_layer=u2, concat_layer=x3_2)

u4 = upconv_concat_RRblock(in_channels=18, out_channels=12, 
    upsize=5)(target_layer=u3, concat_layer=x2_2)

u5 = upconv_concat_RRblock(in_channels=12, out_channels=6, 
    upsize=5)(target_layer=u4, concat_layer=x1_2)

## init mtan block
pre_att_layer = x1_2
pre_target_layer = x1_2
target_layer = x1_2

PS_mtan_E0 = mtan_att_block(in_channels=n_chn[0+1],
    out_channels=n_chn[0+1], stride=1, mode='down')(
        pre_att_layer=pre_att_layer, 
        pre_target_layer=pre_target_layer, 
        target_layer=x1_2
    )

PS_mtan_E1 = mtan_att_block(
    in_channels=n_chn[0+1],out_channels=n_chn[0+2], 
    stride=5, mode='down')(
        pre_att_layer=PS_mtan_E0, 
        pre_target_layer=x1_2, 
        target_layer=x2_2
    )

PS_mtan_E2 = mtan_att_block(
    in_channels=n_chn[0+2],out_channels=n_chn[0+3], 
    stride=5, mode='down')(
        pre_att_layer=PS_mtan_E1, 
        pre_target_layer=x2_2, 
        target_layer=x3_2
    )

PS_mtan_E3 = mtan_att_block(
    in_channels=n_chn[0+3],out_channels=n_chn[0+4], 
    stride=5, mode='down')(
        pre_att_layer=PS_mtan_E2, 
        pre_target_layer=x3_2, 
        target_layer=x4_2
    )

PS_mtan_E4 = mtan_att_block(
    in_channels=n_chn[0+4],out_channels=n_chn[0+5], 
    stride=5, mode='down')(
        pre_att_layer=PS_mtan_E3, 
        pre_target_layer=x4_2, 
        target_layer=x5_2
    )

PS_mtan_E5 = mtan_att_block(
    in_channels=n_chn[0+5],out_channels=n_chn[0+6], 
    stride=5, mode='down')(
        pre_att_layer=PS_mtan_E4, 
        pre_target_layer=x5_2, 
        target_layer=x_bottleneck_2
    )

PS_mtan_D0 = mtan_att_block(
    in_channels=n_chn[0+6], out_channels=n_chn[0+5], 
    stride=5, mode='up')(
        pre_att_layer=PS_mtan_E5, 
        pre_target_layer=x_bottleneck_2, 
        target_layer=u1
    )

PS_mtan_D1 = mtan_att_block(
    in_channels=n_chn[0+5],out_channels=n_chn[0+4], 
     upsize=5, mode='up')(
        pre_att_layer=PS_mtan_D0, 
        pre_target_layer=u1, 
        target_layer=u2
    )

PS_mtan_D2 = mtan_att_block(
    in_channels=n_chn[0+4],out_channels=n_chn[0+3], 
     upsize=5, mode='up')(
        pre_att_layer=PS_mtan_D1, 
        pre_target_layer=u2, 
        target_layer=u3
    )

PS_mtan_D3 = mtan_att_block(
    in_channels=n_chn[0+3],out_channels=n_chn[0+2], 
     upsize=5, mode='up')(
        pre_att_layer=PS_mtan_D2, 
        pre_target_layer=u3, 
        target_layer=u4
    )

PS_mtan_D4 = mtan_att_block(
    in_channels=n_chn[0+2],out_channels=n_chn[0+1], 
     upsize=5, mode='up')(
        pre_att_layer=PS_mtan_D3, 
        pre_target_layer=u4, 
        target_layer=u5
    )