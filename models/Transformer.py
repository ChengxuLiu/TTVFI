import os
import sys
import cv2
import time
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class DynFilter(nn.Module):
    def __init__(self, kernel_size=(3,3), padding=1):
        super(DynFilter, self).__init__()
        self.padding = padding
        filter_localexpand_np = np.reshape(np.eye(np.prod(kernel_size), np.prod(kernel_size)), (np.prod(kernel_size), 1, kernel_size[0], kernel_size[1]))
        self.filter_localexpand = torch.FloatTensor(filter_localexpand_np).cuda()
    def forward(self, x, filter):
        x_localexpand = []
        for c in range(x.size(1)):
            x_localexpand.append(F.conv2d(x[:, c:c + 1, :, :], self.filter_localexpand, padding=self.padding))
        x_localexpand = torch.cat(x_localexpand, dim=1)
        x = torch.sum(torch.mul(x_localexpand, filter), dim=1).unsqueeze(1)
        return x



class GridNet_Filter(nn.Module):
    def __init__(self, input_channel_L1, input_channel_L2, input_channel_L3, output_channel):
        super(GridNet_Filter, self).__init__()

        def First(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1))
            )

        def lateral(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1))
            )

        def downsampling(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=(3, 3), stride=(2, 2),
                                padding=(1, 1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1)),
            )

        def upsampling(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1)),
            )

        def Last(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1))
            )

        self.First_Block1 = First(input_channel_L1, 32)  # 4*RGB(3) + 4* 1st features(32)

        self.Row1_1 = lateral(32, 32)
        self.Row1_2 = lateral(32, 32)
        self.Row1_3 = lateral(32, 32)
        self.Row1_4 = lateral(32, 32)
        self.Row1_5 = lateral(32, 32)
        self.Last_Block11 = Last(32, output_channel) 

        self.Row22_0 = First(input_channel_L2, 64)

        self.Row2_1 = lateral(64, 64)  
        self.Row2_2 = lateral(64, 64)
        self.Row2_3 = lateral(64, 64)
        self.Row2_4 = lateral(64, 64)
        self.Row2_5 = lateral(64, 64)
        self.Last_Block22 = Last(64, output_channel) 

        self.Row33_0 = First(input_channel_L3, 96)

        self.Row3_1 = lateral(96, 96)  
        self.Row3_2 = lateral(96, 96)
        self.Row3_3 = lateral(96, 96)
        self.Row3_4 = lateral(96, 96)
        self.Row3_5 = lateral(96, 96)
        self.Last_Block33 = Last(96, output_channel) 

        self.Col1_1 = downsampling(32, 64)
        self.Col2_1 = downsampling(64, 96)
        self.Col1_2 = downsampling(32, 64)
        self.Col2_2 = downsampling(64, 96)
        self.Col1_3 = downsampling(32, 64)
        self.Col2_3 = downsampling(64, 96)

        self.Col1_4 = upsampling(64, 32)
        self.Col2_4 = upsampling(96, 64)
        self.Col1_5 = upsampling(64, 32)
        self.Col2_5 = upsampling(96, 64)
        self.Col1_6 = upsampling(64, 32)
        self.Col2_6 = upsampling(96, 64)

    def forward(self, Ctx_0t, Ctx_0t_s, Ctx_0, Ctx_1t, Ctx_1t_s, Ctx_1):
        #Ctx_0t, Ctx_1t  bs * 2 * (c+3) * h * w
        Variable1_1 = self.First_Block1(torch.cat((Ctx_0t[0], Ctx_0t_s[0], Ctx_0[0], Ctx_1t[0], Ctx_1t_s[0], Ctx_1[0]), dim=1))  # 1
        Variable1_2 = self.Row1_1(Variable1_1) + Variable1_1  # 2
        Variable1_3 = self.Row1_2(Variable1_2) + Variable1_2  # 3

        Variable2_0 = self.Row22_0(torch.cat((Ctx_0t[1], Ctx_0t_s[1], Ctx_0[1], Ctx_1t[1], Ctx_1t_s[1], Ctx_1[1]), dim=1))  # 4
        Variable2_1 = self.Col1_1(Variable1_1) + Variable2_0  # 5
        Variable2_2 = self.Col1_2(Variable1_2) + self.Row2_1(Variable2_1) + Variable2_1  # 6
        Variable2_3 = self.Col1_3(Variable1_3) + self.Row2_2(Variable2_2) + Variable2_2  # 7

        Variable3_0 = self.Row33_0(torch.cat((Ctx_0t[2], Ctx_0t_s[2], Ctx_0[2], Ctx_1t[2], Ctx_1t_s[2], Ctx_1[2]), dim=1))  # 8
        Variable3_1 = self.Col2_1(Variable2_1) + Variable3_0  # 9
        Variable3_2 = self.Col2_2(Variable2_2) + self.Row3_1(Variable3_1) + Variable3_1  # 10
        Variable3_3 = self.Col2_3(Variable2_3) + self.Row3_2(Variable3_2) + Variable3_2  # 11

        Variable3_4 = self.Row3_3(Variable3_3) + Variable3_3  # 10
        Variable3_5 = self.Row3_4(Variable3_4) + Variable3_4  # 11
        Variable3_6 = self.Row3_5(Variable3_5) + Variable3_5  # 12

        Variable2_4 = self.Col2_4(Variable3_4) + self.Row2_3(Variable2_3) + Variable2_3  # 13
        Variable2_5 = self.Col2_5(Variable3_5) + self.Row2_4(Variable2_4) + Variable2_4  # 14
        Variable2_6 = self.Col2_6(Variable3_6) + self.Row2_5(Variable2_5) + Variable2_5  # 15

        Variable1_4 = self.Col1_4(Variable2_4) + self.Row1_3(Variable1_3) + Variable1_3  # 16
        Variable1_5 = self.Col1_5(Variable2_5) + self.Row1_4(Variable1_4) + Variable1_4  # 17
        Variable1_6 = self.Col1_6(Variable2_6) + self.Row1_5(Variable1_5) + Variable1_5  # 18

        return self.Last_Block11(Variable1_6),self.Last_Block22(Variable2_6),self.Last_Block33(Variable3_6)


class VFIT(BaseNetwork):
    def __init__(self,  num_layer=1, feat_channel=32, patchsize=4, n_head=1, timestep = 0.5, init_weights=False):
        super(VFIT, self).__init__()

        self.timestep = timestep
        self.feat_channel = feat_channel
        self.num_layer = num_layer
        self.Encoder_feat = Feat_Pyramid(out_channel = feat_channel)
        self.Encoder_fusion_L1_Q = nn.Conv2d(32+3, feat_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.Encoder_fusion_L1_KV = nn.Conv2d(32+3, feat_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.Encoder_fusion_L2_Q = nn.Conv2d(32+3, feat_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.Encoder_fusion_L2_KV = nn.Conv2d(32+3, feat_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.Encoder_fusion_L3_Q = nn.Conv2d(32+3, feat_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.Encoder_fusion_L3_KV = nn.Conv2d(32+3, feat_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.FilterNet = GridNet_Filter(6*(3+32),6*(3+32),6*(3+32),(3*3)*6)
        self.Filtering = DynFilter(kernel_size=(3,3), padding=1)

        transformer_L1_list = []
        transformer_L2_list = []
        transformer_L3_list = []
        Cross_Scale_Fusion_list = []
        for i in range(self.num_layer):
            Cross_Scale_Fusion_list.append(Cross_Scale_Fusion(in_channel = feat_channel))
            transformer_L1_list.append(FlowTransformerBlock(patchsize, hidden=feat_channel, n_head=n_head, shift=False if (i % 2 == 0) else True))
            transformer_L2_list.append(FlowTransformerBlock(patchsize, hidden=feat_channel, n_head=n_head, shift=False if (i % 2 == 0) else True))
            transformer_L3_list.append(FlowTransformerBlock(patchsize, hidden=feat_channel, n_head=n_head, shift=False if (i % 2 == 0) else True))
        self.transformer_L1 = nn.ModuleList(transformer_L1_list)
        self.transformer_L2 = nn.ModuleList(transformer_L2_list)
        self.transformer_L3 = nn.ModuleList(transformer_L3_list)
        self.Cross_Scale_Fusion = nn.ModuleList(Cross_Scale_Fusion_list)

        self.Decoder_Cross_Scale_Fusion = Cross_Scale_Fusion(in_channel = feat_channel)
        self.Decoder_feats2 = Feat_Pyramid_Fusion(in_channel = (feat_channel+(32+3)*6))
        if init_weights:
            self.init_weights()

    def Encoder_flow(self, flow):
        flow_pyr = []
        flow_pyr.append(flow)
        for i in range(1, 3):
            flow_pyr.append(F.interpolate(flow, scale_factor=0.5 ** i, mode='bilinear') * (0.5 ** i))
        return flow_pyr

    def Encoder_frame(self, Img):
        img_pyr = []
        img_pyr.append(Img)
        for i in range(1, 3):
            img_pyr.append(F.interpolate(Img, scale_factor=0.5 ** i, mode='bilinear'))
        return img_pyr

    def forward(self,  input_frame, input_flow, input_flow_s):
        # input_flow,input_flow_s # bs * 2*3 * h * w 
        # input_frame bs * 2*3 * h *w
        # extracting features
        b, _, h, w = input_frame.size()

        I0 = input_frame[:, :3, :, :]  # First frame
        I1 = input_frame[:, 3:, :, :]  # Second frame
        Flowt0 = input_flow[:, :2, :, :]  
        Flowt1 = input_flow[:, 2:, :, :]  
        Flowt0_s = input_flow_s[:, :2, :, :] 
        Flowt1_s = input_flow_s[:, 2:, :, :]

        feat_0 = self.Encoder_feat(I0)
        feat_1 = self.Encoder_feat(I1) 
        frame_0 = self.Encoder_frame(I0)
        frame_1 = self.Encoder_frame(I1)
        Flowt0 = self.Encoder_flow(Flowt0)
        Flowt1 = self.Encoder_flow(Flowt1)
        Flowt0_s = self.Encoder_flow(Flowt0_s)
        Flowt1_s = self.Encoder_flow(Flowt1_s)

 
        Embedding_feat_0t  = []
        Embedding_feat_0t_s  = []
        Embedding_feat_1t  = []
        Embedding_feat_1t_s  = []
        Embedding_feat_0  = []
        Embedding_feat_1  = []
        Error_map = []
    
        for i in range(3):
            _,_,hh,ww = frame_0[i].size()

            #generate the consistent token
            Context_0t_s, mask_0t_s = self.warp_with_mask(torch.cat((frame_0[i], feat_0[i]), dim=1), Flowt0_s[i])
            Context_1t_s, mask_1t_s = self.warp_with_mask(torch.cat((frame_1[i], feat_1[i]), dim=1), Flowt1_s[i])
            Context_0t, mask_0t = self.warp_with_mask(torch.cat((frame_0[i], feat_0[i]), dim=1), Flowt0[i])
            Context_1t, mask_1t = self.warp_with_mask(torch.cat((frame_1[i], feat_1[i]), dim=1), Flowt1[i])
            Embedding_feat_0t.append(Context_0t)
            Embedding_feat_0t_s.append(Context_0t_s)
            Embedding_feat_1t.append(Context_1t)
            Embedding_feat_1t_s.append(Context_1t_s)

            #generate the boundary token
            xx = torch.arange(0, ww).view(1, 1, ww).expand(b, hh, ww).float().cuda()# bs * h * w
            yy = torch.arange(0, hh).view(1, hh, 1).expand(b, hh, ww).float().cuda()# bs * h * w 
            Indext0 = (yy + Flowt0_s[i][:,1,:,:]).round().clamp_(0, hh-1) * ww + (xx + Flowt0_s[i][:,0,:,:]).round().clamp_(0, ww-1)# bs * h * w 
            Indext1 = (yy + Flowt1_s[i][:,1,:,:]).round().clamp_(0, hh-1) * ww + (xx + Flowt1_s[i][:,0,:,:]).round().clamp_(0, ww-1)# bs * h * w 
            Indext0 = Indext0.view(b,-1).long()# bs * h*w
            Indext1 = Indext1.view(b,-1).long()# bs * h*w
            Context_0 = torch.cat((frame_0[i], feat_0[i]), dim=1)# bs * 35 * h * w
            Context_1 = torch.cat((frame_1[i], feat_1[i]), dim=1)# bs * 35 * h * w
            Context_0 = Context_0.view(b,3+32,hh,ww).permute(0, 2, 3, 1).contiguous().view(b,hh*ww,35)# b * h*w * c
            Context_1 = Context_1.view(b,3+32,hh,ww).permute(0, 2, 3, 1).contiguous().view(b,hh*ww,35)# b * h*w * c   
            for j in range(b):
                Context_0[j,:,:] = torch.index_select(Context_0[j,:,:],0,Indext0[j,:])
                Context_1[j,:,:] = torch.index_select(Context_1[j,:,:],0,Indext1[j,:])
            Context_0 = Context_0.view(b,hh,ww,3+32).permute(0, 3, 1, 2).contiguous()# bs * 35 * h * w
            Context_1 = Context_1.view(b,hh,ww,3+32).permute(0, 3, 1, 2).contiguous()# bs * 35 * h * w
            Embedding_feat_0.append(Context_0)
            Embedding_feat_1.append(Context_1)

            #generate the confident map
            Emaps = torch.abs(Flowt0_s[i]+Flowt1_s[i]).mean(1, True)
            Emaps = torch.sigmoid(Emaps)*2-1
            Error_map.append(Emaps)
        

#######################################################################################################################        
        Combine_Conv_L1,Combine_Conv_L2,Combine_Conv_L3 = self.FilterNet(Embedding_feat_0t, Embedding_feat_0t_s,Embedding_feat_0, Embedding_feat_1t, Embedding_feat_1t_s,Embedding_feat_1)
        Combine_Conv_L1 = F.softmax(Combine_Conv_L1, dim=1) # bs * (3*3*4) * h * w
        Filtered_input_L1 = []
        for i in range(3+32):
            Filtered_input_L1.append(self.Filtering(torch.cat((Embedding_feat_0t[0][:, i:i + 1, :, :], Embedding_feat_0t_s[0][:, i:i + 1, :, :],Embedding_feat_0[0][:, i:i + 1, :, :],
                                                            Embedding_feat_1t[0][:, i:i + 1, :, :], Embedding_feat_1t_s[0][:, i:i + 1, :, :], Embedding_feat_1[0][:, i:i + 1, :, :]), dim=1), Combine_Conv_L1))
        Image_out = torch.cat(Filtered_input_L1[:3], dim=1)
        Filtered_out_L1 = torch.cat(Filtered_input_L1, dim=1)# 2 * (3+c) * h * w

        Combine_Conv_L2 = F.softmax(Combine_Conv_L2, dim=1) # bs * (3*3*4) * h//2 * w//2
        Filtered_input_L2 = []
        for i in range(3+32):
            Filtered_input_L2.append(self.Filtering(torch.cat((Embedding_feat_0t[1][:, i:i+1, :, :], Embedding_feat_0t_s[1][:, i:i+1, :, :],Embedding_feat_0[1][:, i:i + 1, :, :],
                                                            Embedding_feat_1t[1][:, i:i+1, :, :], Embedding_feat_1t_s[1][:, i:i+1, :, :], Embedding_feat_1[1][:, i:i + 1, :, :]), dim=1), Combine_Conv_L2))
        Filtered_out_L2 = torch.cat(Filtered_input_L2, dim=1)# 2 * (3+c) * h//2 * w//2
        
        Combine_Conv_L3 = F.softmax(Combine_Conv_L3, dim=1) # bs * (3*3*4) * h//4 * w//4
        Filtered_input_L3 = []
        for i in range(3+32):
            Filtered_input_L3.append(self.Filtering(torch.cat((Embedding_feat_0t[2][:, i:i+1, :, :], Embedding_feat_0t_s[2][:, i:i+1, :, :],Embedding_feat_0[2][:, i:i + 1, :, :],
                                                            Embedding_feat_1t[2][:, i:i+1, :, :], Embedding_feat_1t_s[2][:, i:i+1, :, :], Embedding_feat_1[2][:, i:i + 1, :, :]), dim=1), Combine_Conv_L3))
        Filtered_out_L3 = torch.cat(Filtered_input_L3, dim=1)# 2 * (3+c) * h//4 * w//4


        Q_L1 = self.Encoder_fusion_L1_Q(Filtered_out_L1)# bs * feat_c * h * w
        KVw_L1 = torch.stack([Embedding_feat_0t[0],Embedding_feat_0t_s[0],Embedding_feat_1t[0],Embedding_feat_1t_s[0]], dim=1).view(b*4,-1,h,w) # bs*4 * (32+3) * h * w
        KVw_L1 = self.Encoder_fusion_L1_KV(KVw_L1)# bs*2t * feat_c * h * w
        KV_L1 = torch.stack([Embedding_feat_0[0],Embedding_feat_1[0]], dim=1).view(b*2,-1,h,w) # bs*2 * (32+3) * h * w
        KV_L1 = self.Encoder_fusion_L1_KV(KV_L1)# bs * 128 * h * w

        Q_L2 = self.Encoder_fusion_L2_Q(Filtered_out_L2)# bs * feat_c * h//2 * w//2
        KVw_L2 = torch.stack([Embedding_feat_0t[1],Embedding_feat_0t_s[1],Embedding_feat_1t[1],Embedding_feat_1t_s[1]], dim=1).view(b*4,-1,h//2,w//2) # bs*4 * (32+3) * h//2 * w//2
        KVw_L2 = self.Encoder_fusion_L2_KV(KVw_L2)# bs*2t * feat_c * h//2 * w//2
        KV_L2 = torch.stack([Embedding_feat_0[1],Embedding_feat_1[1]], dim=1).view(b*2,-1,h//2,w//2) # bs*2 * (32+3) * h//2 * w//2
        KV_L2 = self.Encoder_fusion_L2_KV(KV_L2)# bs * feat_c * h//2 * w//2
        
        Q_L3 = self.Encoder_fusion_L3_Q(Filtered_out_L3)# bs * feat_c * h//4 * w//4
        KVw_L3 = torch.stack([Embedding_feat_0t[2],Embedding_feat_0t_s[2],Embedding_feat_1t[2],Embedding_feat_1t_s[2]], dim=1).view(b*4,-1,h//4,w//4) # bs*4 * (32+3) * h//4 * w//4
        KVw_L3 = self.Encoder_fusion_L3_KV(KVw_L3)# bs*2t * feat_c * h//4 * w//4
        KV_L3 = torch.stack([Embedding_feat_0[2],Embedding_feat_1[2]], dim=1).view(b*2,-1,h//4,w//4) # bs*2 * (32+3) * h//4 * w//4
        KV_L3 = self.Encoder_fusion_L3_KV(KV_L3)# bs * feat_c * h//4 * w//4

        output_L1 = {'xq': Q_L1, 'xkvw': KVw_L1, 'xkv': KV_L1, 'em':Error_map[0],  'b': b, 't': 2, 'c': Q_L1.size(1)}
        output_L2 = {'xq': Q_L2, 'xkvw': KVw_L2, 'xkv': KV_L2, 'em':Error_map[1],  'b': b, 't': 2, 'c': Q_L2.size(1)}
        output_L3 = {'xq': Q_L3, 'xkvw': KVw_L3, 'xkv': KV_L3, 'em':Error_map[2],  'b': b, 't': 2, 'c': Q_L3.size(1)}
        for i in range(self.num_layer):
            output_L1['xq'],output_L2['xq'],output_L3['xq'] = self.Cross_Scale_Fusion[i](output_L1['xq'],output_L2['xq'],output_L3['xq'])
            output_L1 = self.transformer_L1[i](output_L1)
            output_L2 = self.transformer_L2[i](output_L2)
            output_L3 = self.transformer_L3[i](output_L3)
        output_L1['xq'],output_L2['xq'],output_L3['xq'] = self.Decoder_Cross_Scale_Fusion(output_L1['xq'],output_L2['xq'],output_L3['xq'])

        R_t = self.Decoder_feats2(torch.cat((Embedding_feat_0t[0], Embedding_feat_0t_s[0],Embedding_feat_0[0],Embedding_feat_1t[0], Embedding_feat_1t_s[0],Embedding_feat_1[0],output_L1['xq']), dim=1),
                                torch.cat((Embedding_feat_0t[1], Embedding_feat_0t_s[1],Embedding_feat_0[1],Embedding_feat_1t[1], Embedding_feat_1t_s[1],Embedding_feat_1[1],output_L2['xq']), dim=1),
                                torch.cat((Embedding_feat_0t[2], Embedding_feat_0t_s[2],Embedding_feat_0[2],Embedding_feat_1t[2], Embedding_feat_1t_s[2],Embedding_feat_1[2],output_L3['xq']), dim=1))
        output = Image_out + R_t
        output = torch.clamp(output, 0.0, 1.0)
        return output

#######################################################################################################################

    def warp_with_mask(self, x, flo, return_mask=True):
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
        yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)

        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()

        vgrid = torch.autograd.Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        mask = mask.masked_fill_(mask < 0.999, 0)
        mask = mask.masked_fill_(mask > 0, 1)

        if return_mask:
            return output * mask, mask
        else:
            return output * mask



class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        return self.conv(x)



class Feat_Pyramid(nn.Module):
    def __init__(self,out_channel=32):
        super(Feat_Pyramid, self).__init__()

        self.Feature_First = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))

        self.Feature_Second = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))

        self.Feature_Third = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))

    def forward(self, Input):
        Feature_1 = self.Feature_First(Input)
        Feature_2 = self.Feature_Second(Feature_1)
        Feature_3 = self.Feature_Third(Feature_2)
        return Feature_1, Feature_2, Feature_3


class ResidualBlockNoBN(nn.Module):
    def __init__(self, channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.relu = nn.PReLU()
    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class Feat_Pyramid_Fusion(nn.Module):
    def __init__(self,in_channel=64+35*6):
        super(Feat_Pyramid_Fusion, self).__init__()

        self.Feature_First = nn.Sequential(
            deconv(in_channel, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            )

        self.Feature_Second = nn.Sequential(
            deconv(in_channel+32, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            )

        self.Feature_Third = nn.Sequential(
            nn.Conv2d(in_channel+32, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            ResidualBlockNoBN(64),
            nn.PReLU(),
            ResidualBlockNoBN(64),
            nn.PReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1))


    def forward(self, Input_L1,Input_L2,Input_L3):
        Feature_3 = self.Feature_First(Input_L3)
        Feature_2 = self.Feature_Second(torch.cat([Feature_3,Input_L2],dim=1))
        Feature_1 = self.Feature_Third(torch.cat([Feature_2,Input_L1],dim=1))
        return Feature_1


class Cross_Scale_Fusion(nn.Module):
    def __init__(self, in_channel=32):
        super(Cross_Scale_Fusion,self).__init__()

        self.conv_L1_L2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=2, padding=1),
            nn.PReLU())
        self.conv_L1_L2_L3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=2, padding=1),
            nn.PReLU())

        self.conv_L2_L1 = nn.Sequential(
            deconv(in_channel, in_channel, kernel_size=3, padding=1),
            nn.PReLU())
        self.conv_L2_L3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=2, padding=1),
            nn.PReLU())

        self.conv_L3_L2 = nn.Sequential(
            deconv(in_channel, in_channel, kernel_size=3, padding=1),
            nn.PReLU())
        self.conv_L3_L2_L1 = nn.Sequential(
            deconv(in_channel, in_channel, kernel_size=3, padding=1),
            nn.PReLU())

        self.Merge_L1 = nn.Sequential(
            nn.Conv2d(3 * in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.PReLU())
        self.Merge_L2 = nn.Sequential(
            nn.Conv2d(3 * in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.PReLU())
        self.Merge_L3 = nn.Sequential(
            nn.Conv2d(3 * in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.PReLU())

    def forward(self, x1, x2, x3):

        x12 = self.conv_L1_L2(x1)
        x13 = self.conv_L1_L2_L3(x12)

        x21 = self.conv_L2_L1(x2)
        x23 = self.conv_L2_L3(x2)

        x32 = self.conv_L3_L2(x3)
        x31 = self.conv_L3_L2_L1(x32)

        x1 = self.Merge_L1( torch.cat((x1, x21, x31), dim=1) )
        x2 = self.Merge_L2( torch.cat((x2, x12, x32), dim=1) )
        x3 = self.Merge_L3( torch.cat((x3, x13, x23), dim=1) )
        
        return x1, x2, x3


# #############################################################################
# ############################# Transformer  ##################################
# #############################################################################

class MultiHeadedAttention2(nn.Module):
    """
    Take in model size and number of heads.
    """
    def __init__(self, patchsize=8, d_model=35, n_head=1,RPE = False):
        super().__init__()
        self.patchsize = patchsize
        self.n_head = n_head
        self.RPE = RPE
        self.scale = d_model ** -0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, kv, b, t, c):
        # Q  b*(h//k)*(w//k) * 1*k*k * c
        # KV  b*(h//k)*(w//k) * t*k*k * c
        # mask  b*(h//k)*(w//k) * 1*k*k * 1
        b, nkv, c = kv.size()
        nq = q.size(1)
        sc = c//self.n_head
        q = q.reshape(b, nq, self.n_head, sc).permute(0, 2, 1, 3) # b * n_head * nq * sc
        kv = kv.reshape(b, nkv, self.n_head, sc).permute(0, 2, 1, 3) # b * n_head * nk * sc
        q = q * self.scale  # b * n_head * nq * sc
        attn = (q @ kv.transpose(-2, -1))# b * n_head * nq * nkv
        attn = self.softmax(attn)# b * n_head * nq * nk
        x = (attn @ kv)# b * n_head * nq * sc
        x = x.transpose(1, 2).reshape(b, nq, c)# b * nq * c
        # b*(h//k)*(w//k) * t*k*k * c
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, T, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, T, window_size, window_size, C)
    """
    B, T, H, W, C = x.shape
    x = x.view(B, T, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(-1, T, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (b*(h//k)*(w//k) * t * k * k * c)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, T, H, W, C)
    """
    wb, t, _, _, c = windows.shape
    B = int(wb / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, t, window_size, window_size, c)
    x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(B, t, H, W, c)
    return x



# #############################################################################
# ############################# Flow Transformer  #############################
# #############################################################################


class FlowTransformerBlock(nn.Module):
    def __init__(self, patchsize=8, hidden=128, n_head=1, shift=False):
        super().__init__()
        self.shift = shift
        self.shift_size = patchsize//2
        self.patchsize = patchsize
        self.embedding_Q = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1),
            nn.PReLU())
        self.embedding_KV = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1),
            nn.PReLU())
        self.multiheadattention = MultiHeadedAttention2(patchsize = patchsize, d_model=hidden, n_head=n_head)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1),
            nn.PReLU())


    def forward(self, x):
        Q, KVw, KV, emask, b, t, c = x['xq'],x['xkvw'],x['xkv'],x['em'], x['b'], x['t'], x['c']
        # Q      b * c * h * w
        # Kw,Vw      b*4 * c * h * w
        # K,V      b*2 * c * h * w
        # emask     b * 1 * h * w
        h, w = Q.size(2), Q.size(3) 
        Q = Q +  self.embedding_Q(Q)# b * c * h * w  
        KV = KV + self.embedding_KV(KV) # b*2 * c * h * w
        KVw = KVw + self.embedding_KV(KVw) # b*4 * c * h * w

        shortcut = Q# b * c * h * w
        Q = Q.view(b,1,c,h,w).permute(0, 1, 3, 4, 2).contiguous()# b * 1 * h * w * c
        KVws = KVw.view(b,2*t,c,h,w).permute(0, 1, 3, 4, 2).contiguous() # b * 2t * h * w * c
        KVs = KV.view(b,t,c,h,w).permute(0, 1, 3, 4, 2).contiguous() # b * t * h * w * c
        Masks = emask.view(b,1,-1,h,w).permute(0, 1, 3, 4, 2).contiguous() # b * 1 * h * w * 1
        if self.shift:
            shifted_Q = torch.roll(Q, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            KVs = torch.roll(KVs, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            KVws = torch.roll(KVws, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            # Masks = torch.roll(Masks.int(), shifts=(-self.shift_size, -self.shift_size), dims=(2, 3)).eq(1)
            Masks = torch.roll(Masks, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_Q = Q # b * t * h * w * c
            KVs = KVs # b * t * h * w * c
            KVws = KVws # b * t * h * w * c
            Masks = Masks # b * 1 * h * w * 1
        shifted_Q = window_partition(shifted_Q, self.patchsize)  # b * t * h * w * c  -->   b*(h//k)*(w//k) * 1 * k * k * c
        KVs = window_partition(KVs, self.patchsize)  # b * t * h * w * c  -->   b*(h//k)*(w//k) * t * k * k * c
        KVws = window_partition(KVws, self.patchsize)  # b * 2*t * h * w * c  -->   b*(h//k)*(w//k) * 2*t * k * k * c
        Masks = window_partition(Masks, self.patchsize)  # b * 1 * h * w * 1  -->   b*(h//k)*(w//k) * 1 * k * k * 1
        wb = shifted_Q.size(0) # b*(h//k)*(w//k) * 1 * k * k * c
        shifted_Q = shifted_Q.view(wb, -1, c) # b*(h//k)*(w//k) * 1*k*k * c
        KVs = KVs.view(wb, -1, c) # b*(h//k)*(w//k) * t*k*k * c
        KVws = KVws.view(wb, -1, c) # b*(h//k)*(w//k) * 2*t*k*k * c
        Masks = Masks.view(wb, -1, 1) # b*(h//k)*(w//k) * 1*k*k * 1
        # shifted_Q = self.multiheadattention(shifted_Q, KVs, Masks, b, t, c)+ self.multiheadattention(shifted_Q, KVws, Masks, b, 2*t, c) # b*(h//k)*(w//k) * 1*k*k * c
        shifted_Q = Masks * self.multiheadattention(shifted_Q, KVs, b, t, c) + (1 - Masks) *self.multiheadattention(shifted_Q, KVws, b, 2*t, c) # b*(h//k)*(w//k) * 1*k*k * c
        shifted_Q = shifted_Q.view(-1, 1, self.patchsize, self.patchsize, c) # b*(h//k)*(w//k) * 1 * k * k * c
        shifted_Q = window_reverse(shifted_Q, self.patchsize, h, w)  # B, T, H, W, C
        if self.shift:
            Q = torch.roll(shifted_Q, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            Q = shifted_Q
        # FFN
        Q = shortcut + Q.permute(0,1,4,2,3).contiguous().view(b,c,h,w)  # B, C, H, W
        Q = Q + self.feed_forward(Q)
        return {'xq': Q, 'xkvw': KVw, 'xkv': KV, 'em': emask,  'b': b, 't': t, 'c': c}


