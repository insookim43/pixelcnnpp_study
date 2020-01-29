import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

class PixelCNNLayer_down(nn.Module):
    def __init__(nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet

        # u_stream : pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters,
        down_shifted_conv2d, resnet_nonlinearity, skip_connection=1)
        for _ in range(nr_resnet)])
        
        # ul_stream : pixels above and to left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters,
        down_right_shifted_conv2d, resnet_nonlinearity, skip_connection=2)
        for _ in range(nr_resnet)])

    
    def forward(self, u, ul, u_list, ul_list):
        for i in range(self.nr_resnet):
            u = self.u_stream[i](u, a=u_list.pop())
            ul = self.ul.stream[i](ul, a = torch.cat((u, ul_list.pop()),1))

class PixelCNNLayer_up(nn.Module):
    def __init__(nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        
        # u_stream : stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters,
        down_shifted_conv2d, resnet_nonlinearity, skip_connection = 0)
        for _ in range(nr_resnet)])
        
        # ul_stream : stream from pixels above and to the left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters,
        down_right_shifted_conv2d, resnet_nonlinearity, skip_connection = 1)
        for _ in range(nr_resnet)])
    def forward(self, u, ul):
        u_list, ul_list = [], []

        for i in range(self.nr_resnet):
        u = self.u_stream[i](u)
        ul = self.ul_stream[i](ul, a=u)
        u_list += [u]
        ul_list += [ul]

        return u_list, ul_list
    

class PixelCNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
    resnet_nonlinearity = 'concat_elu', input_channels=3):    
        super(PixelCNN, self).__init__()
        self.resnet_nonlinearity = lambda x : concat_elu(x)

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        
        down_nr_resnet = [5, 6, 6] # [nr_resnet] + [nr_resnet + 1 ] * 2

        # down resnet layers
        self.down_layer_slice1 = PixelCNNLayer_down(down_nr_resnet[0],
        nr_filters, self.resnet_nonlinearity)
        self.down_layer_slice2 = PixelCNNLayer_down(down_nr_resnet[1],
        nr_filters, self.resnet_nonlinearity)
        self.down_layer_slice3 = PixelCNNLayer_down(down_nr_resnet[2],
        nr_filters, self.resnet_nonlinearity)

        # up resnet layers
        self.up_layer_slice1 = PixelCNN_Layer_up(nr_resnet, nr_filters,
        self.resnet.resnet_nonlinearity)
        self.up_layer_slice2 = PixelCNN_Layer_up(nr_resnet, nr_filters,
        self.resnet.resnet_nonlinearity)    
        self.up_layer_slice3 = PixelCNN_Layer_up(nr_resnet, nr_filters,
        self.resnet.resnet_nonlinearity)

        # down shifted conv2d
        self.downsize_u_stream_slice1 = down_shifted_conv2d(nr_filters, 
        nr_filters, stride=(2,2))
        self.downsize_u_stream_slice2 = down_shifted_conv2d(nr_filters, 
        nr_filters, stride=(2,2))

        # down right shifted conv2d
        self.downsize_ul_stream_slice1 = down_right_shifted_conv2d(
            nr_filters, nr_filters, stride=(2,2))
        self.downsize_ul_stream_slice2 = down_right_shifted_conv2d(
            nr_filters, nr_filters, stride=(2,2))


        # down shifted deconv2d
        self.upsize_u_stream_slice1 = down_shifted_deconv2d(nr_filters,
        nr_filters, stride=(2,2))
        self.upsize_u_stream_slice2 = down_shifted_deconv2d(nr_filters,
        nr_filters, stride=(2,2))

        # down right shifted deconv2d
        self.upsize_ul_stream_slice1 = down_right_shifted_deconv2d(nr_filters,
        nr_filters, stride=(2,2))
        self.upsize_ul_stream_slice1 = down_right_shifted_deconv2d(nr_filters,
        nr_filters, stride=(2,2))


        # initial down_shifted_conv2d
        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters,
        filter_size=(2,3), shift_output_down=True)

        # initial down_shifted_conv2d & down_right_shifted_conv2d
        self.ul_init_slice1 = down_shifted_conv2d(input_channels + 1,
        nr_filters, filter_size=(1,3), shift_output_down =True)
        self.ul_init_slice2 = down_right_shifted_conv2d(input_channels + 1,
        nr_filters, filter_size=(2,1), shift_output_down =True)

        num_mix = 3 if self.input_channels ==1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None


    def forward(self, x, sample=False):

        # input initialize and padding
        # (train or sample)
        if sample :
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad = False)
            padding = padding.cuda() if x.is_cuda else padding
            x = torch.cat((x, padding), 1)

        if self.init_padding is None and not sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.cuda() if x.is_cuda else padding


        # network input ## up pass ##

        if sample :
            x = x 
        else : 
            x = torch.cat((x, self.init_padding), 1)

        u_list = [self.u_init(x)]
        ul_list = [self.ul_init_slice1(x) + self.ul_init_slice2(x)]

        # resnet block
        u_out_0, ul_out_0 = self.up_layer_slice1(u_list[-1], ul_list[-1])
        u_list += u_out_0
        ul_list += ul_out_0
        #downscale
        u_list += [self.downsize_u_stream_slice1(u_list[-1])]
        ul_list += [self.downsize_ul_stream_slice1(ul_list[-1])]

        # resnet block
        u_out_1, ul_out_1 = self.up_layer_slice2(u_list[-1], ul_list[-1])
        u_list += u_out_1
        ul_list += ul_out_1
        u_list += [self.downsize_u_stream_slice2(u_list[-1])]
        ul_list += [self.downsize_ul_stream_slice2(ul_list[-1])]

        #downscale
        u_out_2, ul_out_2 = self.up_layer_slice2(u_list[-1], ul_list[-1])
        u_list += u_out_2
        ul_list += ul_out_2

        # Down Pass
        u = u_list.pop()
        ul = ul_list.pop()

        # resnet block
        u, ul = self.down_layer_slice1(u, ul_u_list, ul_list)
        # upscale 
        u = self.upsize_u_stream_slice1(u)
        ul = self.upsize_ul_stream_slice1(ul)

        #resnet block
        u, ul = self.down_layer_slice2(u, ul_u_list, ul_list)
        # upscale 
        u = self.upsize_u_stream_slice2(u)
        ul = self.upsize_ul_stream_slice2(ul)

        #resnet block
        u, ul = self.down_layer_slice3(u, ul_u_list, ul_list)

        x_out = self.nin_out(F.elu(ul))
        
        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out