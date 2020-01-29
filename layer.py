import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

class PixelCNNLayer_down(nn.Module):
    def __init__(nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNN, self).__init__()
    
    def forward(self, u, ul, u_list, ul_list):
        for i in range(self.nr_resnet):
            u = self.u_stream[i](u, a=u_list.pop())
            ul = self.ul.stream[i](ul, a = torch.cat((u, ul_list.pop()),1))

    

class PixelCNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
    resnet_nonlinearity = 'concat_elu', input_channels=3):    
        super(PixelCNN, self).__init__()
        self.resnet_nonlinearity = lambda x : concat_elu(x)

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        
        down_nr_resnet = [5, 6, 6] # [nr_resnet] + [nr_resnet + 1 ] * 2

        self.down_layer_slice1 = PixelCNNLayer_down(down_nr_resnet[0],
        nr_filters, self.resnet_nonlinearity)

        self.down_layer_slice2 = PixelCNNLayer_down(down_nr_resnet[1],
        nr_filters, self.resnet_nonlinearity)

        self.down_layer_slice3 = PixelCNNLayer_down(down_nr_resnet[2],
        nr_filters, self.resnet_nonlinearity)

        self.up_layer_slice1 = PixelCNN_Layer_up

    def forward():
        pass

