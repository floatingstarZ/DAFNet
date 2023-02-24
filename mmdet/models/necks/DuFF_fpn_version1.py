import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from copy import deepcopy
from ..builder import NECKS



def weight_init(m):
    # weight_initialization: important for wgan
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0, 0.02)
    elif class_name.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)


#     else:print(class_name)

#################################
# class Dummy(BaseModule):
#     """
#     https://github.com/zyjwuyan/BBS-Net/blob/master/models/BBSNet_model.py
#     """
#
#     def __init__(self, in_planes):
#         super(Dummy, self).__init__()
#
#     def forward(self, x):
#         return dict(atte_map=x.new_zeros(x.shape))
#
#
# class SpatialAttention(BaseModule):
#     """
#     https://github.com/zyjwuyan/BBS-Net/blob/master/models/BBSNet_model.py
#                              loc='x',
#                          type='Spatial',
#                          atte_type='Max',
#                          use_sigmoid=False,
#                          use_proj=True,
#                          proj_norm=True
#     """
#
#     def __init__(self,
#                  in_planes):
#         super(SpatialAttention, self).__init__()
#         # spatial feature map
#         self.conv1 = nn.Conv2d(1, 1, 1)
#         device = self.conv1.weight.data.device
#         self.conv1.weight.data = torch.ones(1)[:, None, None, None].to(device)
#         self.conv1.bias.data = torch.rand(1).to(device)
#
#     def forward(self, x):
#         outs = dict()
#
#         # spatial feature map
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         atte_map = max_out
#         atte_map = self.conv1(atte_map)
#         outs['atte_map'] = atte_map
#
#         return outs
#
#
# class ChannelAttention(BaseModule):
#     """
#     https://github.com/zyjwuyan/BBS-Net/blob/master/models/BBSNet_model.py
#     """
#
#     def __init__(self,
#                  in_planes):
#         super(ChannelAttention, self).__init__()
#         # GAP
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         # semantic feature map
#         self.conv1 = nn.Conv2d(in_planes, in_planes, 1)
#         device = self.conv1.weight.data.device
#
#         init_U = torch.randn(in_planes, in_planes) * 0.1
#         init_U = init_U / torch.norm(init_U, dim=1, keepdim=True)
#         U = init_U[:, :, None, None]
#         self.conv1.weight.data = U.to(device)
#         # self.conv1.bias.data = torch.randn(in_planes).to(device)
#
#
#     def forward(self, x):
#         outs = dict()
#         # semantic feature map
#         atte_map = self.gap(x)
#         atte_map = self.conv1(atte_map)
#
#         # orthogonal regularization
#         U = self.conv1.weight.data[:, :, 0, 0]
#         utu = torch.matmul(U, U.transpose(1, 0))
#         target = torch.eye(len(U), device=x.device)
#         og_loss = torch.dist(utu, target, p=2)
#
#         outs['loss'] = og_loss
#         outs['atte_map'] = atte_map
#         return outs
#
# class L2_Loss(nn.Module):
#     def __init__(self, gamma=1, avg=False):
#         super(L2_Loss, self).__init__()
#         self.gamma = gamma
#         self.avg = avg
#
#     def forward(self, feat_org, feat_fusion):
#         if not self.avg:
#             return torch.dist(feat_org, feat_fusion, p=2) * self.gamma
#         else:
#             loss = torch.mean((feat_org - feat_fusion) ** 2)
#             return loss * self.gamma
#
#
# class Cos_Loss(nn.Module):
#     def __init__(self, gamma=1):
#         super(Cos_Loss, self).__init__()
#         self.gamma = gamma
#
#     def forward(self, feat_org, feat_fusion):
#         loss = torch.nn.CosineSimilarity(dim=1)(feat_org, feat_fusion)
#         return torch.mean(loss) * self.gamma
#
# class AFF(BaseModule):
#
#     def __init__(self,
#                  planes,
#                  ex_planes,
#                  rebuild_feat=[
#                      dict(
#                          loc='x',
#                          type='Channel',
#                      ),
#                      dict(
#                          loc='x',
#                          type='Spatial',
#                      ),
#                  ],
#                  rebuild_loss=[
#                      dict(
#                          type='L2',
#                          gamma=0.01
#                      ),
#                      dict(
#                          type='L2',
#                          gamma=0.001
#                      ),
#                  ],
#                  contrast_feat=dict(
#                      type='Channel',
#                  ),
#                  contrast_loss=dict(
#                      type='L2',
#                      gamma=0.01
#                  ),
#                  alpha=1,
#                  beta=1,
#                  fusion_strategy=0,
#                  init_cfg=None):
#         super(AFF, self).__init__(init_cfg)
#         self.planes = planes
#         self.ex_planes = ex_planes
#         self.alpha = alpha
#         self.beta = beta
#         r_feat_cfgs = deepcopy(rebuild_feat)
#         r_loss_cfgs = deepcopy(rebuild_loss)
#         self.rebuilds = nn.ModuleList()
#         self.rebuild_src = [feat_cfg.pop('loc')
#                             for feat_cfg in r_feat_cfgs]
#         self.fusion_strategy = fusion_strategy
#         for feat_cfg, loss_cfg in zip(r_feat_cfgs, r_loss_cfgs):
#             rebuild_dict = nn.ModuleDict()
#             rebuild_dict['feat_org'] = self.build_feat(feat_cfg, planes)
#             rebuild_dict['feat_fusion'] = self.build_feat(feat_cfg, planes)
#             rebuild_dict['loss'] = self.build_loss(loss_cfg)
#             self.rebuilds.append(rebuild_dict)
#
#         self.cf1 = self.build_feat(contrast_feat, planes)
#         self.cf2 = self.build_feat(contrast_feat, planes)
#         self.loss_c = self.build_loss(contrast_loss)
#
#         self.trans_conv1 = nn.Conv2d(planes, planes,
#                                      3, 1, 1, bias=True)
#         self.trans_conv2 = nn.Conv2d(planes, planes,
#                                      3, 1, 1, bias=True)
#         self.trans_conv1.apply(weight_init)
#         self.trans_conv2.apply(weight_init)
#
#     def build_feat(self, feat_cfg, planes):
#         f_cfg = deepcopy(feat_cfg)
#         f_type = f_cfg.pop('type') if f_cfg else None
#         if f_type == 'Spatial':
#             feat_module = SpatialAttention
#             f_out_channel = 1
#         elif f_type == 'Channel':
#             feat_module = ChannelAttention
#             f_out_channel = planes
#         elif f_type == 'Dummy':
#             feat_module = Dummy
#             f_out_channel = planes
#         else:
#             feat_module = Dummy
#             f_out_channel = planes
#         if f_cfg:
#             feat = feat_module(f_out_channel, **f_cfg)
#         else:
#             feat = feat_module(f_out_channel)
#         return feat
#
#     def build_loss(self, loss_cfg):
#         l_cfg = deepcopy(loss_cfg)
#         assert l_cfg
#         l_type = l_cfg.pop('type')
#
#         if l_type == 'L2':
#             loss_module = L2_Loss
#         # elif l_type == 'Cos':
#         #     loss_module = Cos_Loss
#         else:
#             raise Exception('Loss type: %s undefined' % l_type)
#         loss = loss_module(**l_cfg)
#         return loss
#
#     def forward(self, x, x_ex, return_loss=False):
#         """Forward function."""
#         x_input = x
#         x_ex_input = x_ex
#         loss = x.new_zeros(1)
#         org_rfs = []
#         for src, rebuild_dict in zip(self.rebuild_src,
#                                      self.rebuilds):
#             if src == 'x':
#                 feat = x
#             elif src == 'x_ex':
#                 feat = x_ex
#             else:
#                 raise Exception('Wrong type: %s' % src)
#             feat_out = rebuild_dict['feat_org'](feat)
#             if 'loss' in feat_out.keys():
#                 loss = loss + feat_out['loss']
#             org_rfs.append(feat_out['atte_map'])
#
#         x = self.trans_conv1(x)
#         x_ex = self.trans_conv2(x_ex)
#         cf1_out = self.cf1(x)
#         cf2_out = self.cf2(x_ex)
#         cf1 = cf1_out['atte_map']
#         cf2 = cf2_out['atte_map']
#         if 'loss' in cf1_out.keys():
#             loss = loss + cf1_out['loss']
#         if 'loss' in cf2_out.keys():
#             loss = loss + cf2_out['loss']
#         # output the fusion feature
#         if self.fusion_strategy == 0:
#             x_fusion = x + x_ex + x_input
#         else:
#             raise Exception('Wrong fusion strategy:')
#         for src, org_rf, rebuild_dict in zip(self.rebuild_src,
#                                              org_rfs,
#                                              self.rebuilds):
#
#             feat_out = rebuild_dict['feat_fusion'](x_fusion)
#             fusion_rf = feat_out['atte_map']
#             if 'loss' in feat_out.keys():
#                 loss = loss + feat_out['loss']
#             loss_rebuild = rebuild_dict['loss'](org_rf, fusion_rf)
#             if src == 'x':
#                 loss = loss + loss_rebuild
#             elif src == 'x_ex':
#                 loss = loss + self.alpha * loss_rebuild
#             else:
#                 raise Exception('Wrong type: %s' % src)
#
#         loss_c = self.loss_c(cf1, cf2)
#
#         loss = loss + self.beta * loss_c
#         if return_loss:
#             # print(loss)
#             return loss, x_fusion
#         else:
#             return x_fusion
#################################

class Dummy(BaseModule):
    """
    https://github.com/zyjwuyan/BBS-Net/blob/master/models/BBSNet_model.py
    """

    def __init__(self, in_planes):
        super(Dummy, self).__init__()

    def forward(self, x):
        return dict(atte_map=x.new_zeros(x.shape))


class SpatialAttention(BaseModule):
    """
    https://github.com/zyjwuyan/BBS-Net/blob/master/models/BBSNet_model.py
    """

    def __init__(self, in_planes,
                 ds_ratio=1,
                 atte_type='V1',
                 use_sigmoid=True,
                 use_proj=False,
                 proj_norm=False):
        super(SpatialAttention, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.sigmoid = nn.Sigmoid()
        self.ds_ratio = ds_ratio
        self.proj_norm = proj_norm
        self.use_proj = use_proj

        self.atte_type = atte_type
        if self.atte_type == 'V1':
            self.conv1 = nn.Conv2d(1, 1, 3, padding=1, bias=False)
            self.conv1.apply(weight_init)
        elif self.atte_type in ['Max', 'Mean']:
            if self.use_proj:
                # self.w = nn.Parameter(torch.ones(1))
                # self.b = nn.Parameter(torch.rand(1))
                self.conv1 = nn.Conv2d(1, 1, 1)
                device = self.conv1.weight.data.device
                self.conv1.weight.data = torch.ones(1)[:, None, None, None].to(device)
                self.conv1.bias.data = torch.rand(1).to(device)

    def forward(self, x):
        B, C, H, W = x.shape
        H_ds = max(H // self.ds_ratio, 1)
        W_ds = max(W // self.ds_ratio, 1)
        outs = dict()
        x = nn.AdaptiveAvgPool2d((H_ds, W_ds))(x)
        if self.atte_type == 'V1':
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            atte_map = self.conv1(max_out)
        elif self.atte_type == 'Max':
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            atte_map = max_out
            if self.use_proj:
                atte_map = self.conv1(atte_map)
                # atte_map = self.w * atte_map + self.b
        elif self.atte_type == 'Mean':
            mean_out = torch.mean(x, dim=1, keepdim=True)
            atte_map = mean_out
            if self.use_proj:
                atte_map = self.conv1(atte_map)
                # atte_map = self.w * atte_map + self.b
        if self.use_sigmoid:
            atte_map = self.sigmoid(atte_map)
        outs['atte_map'] = atte_map
        if self.proj_norm and self.use_proj:
            outs['loss'] = torch.abs(self.w - x.new_ones(1))
        # print(self.conv1.weight)
        return outs


class ChannelAttention(BaseModule):
    """
    https://github.com/zyjwuyan/BBS-Net/blob/master/models/BBSNet_model.py
    """

    def __init__(self, in_planes,
                 atte_type='V1',
                 ratio=4,
                 use_sigmoid=True,
                 use_proj=False,
                 proj_norm=True):
        super(ChannelAttention, self).__init__()
        self.atte_type = atte_type
        self.use_sigmoid = use_sigmoid
        self.use_proj = use_proj
        self.proj_norm = proj_norm
        if self.atte_type == 'V1':
            self.max_pool = nn.AdaptiveMaxPool2d(1)

            self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

            self.sigmoid = nn.Sigmoid()
            self.fc1.apply(weight_init)
            self.fc2.apply(weight_init)
        elif self.atte_type == 'Linear':
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.sigmoid = nn.Sigmoid()
            if self.use_proj:
                #-----Implementation of matrix multiplication
                # init_U = torch.randn(in_planes, in_planes) * 0.1
                # init_U = init_U / torch.norm(init_U, dim=1, keepdim=True)
                # self.U = nn.Parameter(init_U)
                # self.bias = nn.Parameter(torch.randn(in_planes))

                # -----Implementation of Convolution layer
                self.conv1 = nn.Conv2d(in_planes, in_planes, 1, bias=False)
                device = self.conv1.weight.data.device
                init_U = torch.randn(in_planes, in_planes) * 0.1
                init_U = init_U / torch.norm(init_U, dim=1, keepdim=True)
                U = init_U[:, :, None, None]
                self.conv1.weight.data = U.to(device)
                # self.conv1.bias.data = torch.randn(in_planes).to(device)


        elif self.atte_type == 'Max':
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        outs = dict()
        if self.atte_type == 'V1':
            atte_map = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        elif self.atte_type == 'Linear':
            # ----- Before max
            atte_map = self.max_pool(x)
            if self.use_proj:
                #-----Implementation of matrix multiplication
                # x = x.reshape(B, C, H * W)
                # U = self.U[None, ...].expand(B, C, C)
                # b = self.bias[None, ...].expand(B, C)[..., None]
                # atte_map = torch.matmul(U, x)#  + b
                # atte_map = atte_map.reshape(B, C, H, W)
                # -----Implementation of Convolution layer
                atte_map = self.conv1(x)

                if self.proj_norm:
                    #-----Implementation of Convolution layer
                    U = self.conv1.weight[:, :, 0, 0]
                    utu = torch.matmul(U, U.transpose(1, 0))
                    target = torch.eye(len(U), device=x.device)
                    og_loss = torch.dist(utu, target, p=2)
                    outs['loss'] = og_loss
                    #-----Implementation of matrix multiplication
                    # utu = torch.matmul(self.U, self.U.transpose(1, 0))
                    # target = torch.eye(len(self.U), device=x.device)
                    # F2norm = torch.dist(utu, target, p=2)
                    # outs['loss'] = F2norm
            else:
                atte_map = x
            # ----- After max
            # atte_map = self.max_pool(atte_map)

        elif self.atte_type == 'Max':
            atte_map = self.max_pool(x)
        if self.use_sigmoid:
            atte_map = self.sigmoid(atte_map)
        outs['atte_map'] = atte_map
        return outs


class Compress(BaseModule):
    """
    https://github.com/zyjwuyan/BBS-Net/blob/master/models/BBSNet_model.py
    """

    def __init__(self, in_planes, ratio=16):
        super(Compress, self).__init__()
        self.compress_conv = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 3, 1, 1),
            nn.BatchNorm2d(in_planes // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes // ratio, 3, 1, 1),
            nn.BatchNorm2d(in_planes // ratio),
            nn.ReLU(inplace=True)
        )
        self.compress_conv.apply(weight_init)

    def forward(self, x):
        feat = self.compress_conv(x)
        return feat


def attention(query, key):
    """

    :param query: B x n_Q x E
    :param key:   B x n_K x E
    :param value: B x n_V(n_K) x E
    :param mask:
    :param dropout:
    :return:
    """
    # "Compute 'Scaled Dot Product Attention'"
    # n_Q x E x E matmul E x  n_K -> n_Q x n_K
    scores = torch.matmul(query, key.transpose(-2, -1))
    # B x n_Q x n_K
    p_attn = scores.softmax(dim=-1)
    return p_attn


class Correlation(BaseModule):
    # from mmcv.cnn.bricks
    def __init__(self,
                 in_planes,
                 ds_ratio=1,
                 init_cfg=None):
        super(Correlation, self).__init__(init_cfg)
        # self.q = nn.Linear(in_planes, in_planes, bias=True)
        # self.k = nn.Linear(in_planes, in_planes, bias=True)
        self.ds_ratio = ds_ratio

    def forward(self, x):
        outs = dict()

        B, C, H, W = x.shape
        H_ds = max(H // self.ds_ratio, 1)
        W_ds = max(W // self.ds_ratio, 1)
        x = nn.AdaptiveAvgPool2d((H_ds, W_ds))(x)
        B, C, H, W = x.shape
        # B x nC x H*W
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        k = x
        q = x
        # k = self.k(x)
        # q = self.q(x)
        corr_map = attention(q, k).reshape(B, 1, H * W, H * W)
        outs['atte_map'] = corr_map
        return outs


class Gram(BaseModule):
    # from mmcv.cnn.bricks
    def __init__(self,
                 in_planes,
                 ds_ratio=1,
                 init_cfg=None):
        super(Gram, self).__init__(init_cfg)
        # self.q = nn.Linear(in_planes, in_planes, bias=True)
        # self.k = nn.Linear(in_planes, in_planes, bias=True)
        self.ds_ratio = ds_ratio

    def forward(self, x):
        outs = dict()
        B, C, H, W = x.shape
        H_ds = max(H // self.ds_ratio, 1)
        W_ds = max(W // self.ds_ratio, 1)
        x = nn.AdaptiveAvgPool2d((H_ds, W_ds))(x)
        B, C, H, W = x.shape
        # B x nC x H*W
        x = x.reshape(B, C, H * W)
        gram = torch.matmul(x, x.transpose(-2, -1))
        outs['atte_map'] = gram
        return outs


class Discriminator_Loss(nn.Module):
    def __init__(self, in_planes,
                 ratio=4,
                 layer=2,
                 gamma=1):
        super(Discriminator_Loss, self).__init__()
        self.gamma = gamma
        self.DNet = nn.ModuleList()
        inner_planes = max(in_planes // ratio, 1)
        self.DNet.append(nn.Sequential(
            nn.Conv2d(in_planes, inner_planes, 3, 1, 1),
            nn.BatchNorm2d(inner_planes),
            nn.ReLU(inplace=True)
        ))
        for i in range(layer):
            self.DNet.append(
                nn.Sequential(
                    nn.Conv2d(inner_planes,
                              inner_planes, 3, 1, 1),
                    nn.BatchNorm2d(inner_planes),
                    nn.ReLU(inplace=True)
                )
            )
        self.DNet.append(
            nn.Sequential(
                nn.Conv2d(inner_planes,
                          1, 3, 1, 1),
            )
        )
        self.DNet = nn.Sequential(*self.DNet)
        self.DNet.apply(weight_init)

    def forward(self, feat_org, feat_fusion):
        pred_org = self.DNet(feat_org)
        pred_fusion = self.DNet(feat_fusion)
        d_loss = -torch.mean(pred_org) + torch.mean(pred_fusion) + \
                 torch.dist(feat_org, feat_fusion, p=2)
        return d_loss * self.gamma


class L2_Loss(nn.Module):
    def __init__(self, gamma=1, avg=False):
        super(L2_Loss, self).__init__()
        self.gamma = gamma
        self.avg = avg

    def forward(self, feat_org, feat_fusion):
        if not self.avg:
            return torch.dist(feat_org, feat_fusion, p=2) * self.gamma
        else:
            loss = torch.mean((feat_org - feat_fusion) ** 2)
            return loss * self.gamma


class Cos_Loss(nn.Module):
    def __init__(self, gamma=1):
        super(Cos_Loss, self).__init__()
        self.gamma = gamma

    def forward(self, feat_org, feat_fusion):
        loss = torch.nn.CosineSimilarity(dim=1)(feat_org, feat_fusion)
        return torch.mean(loss) * self.gamma


class AFF(BaseModule):

    def __init__(self,
                 planes,
                 ex_planes,
                 rebuild_feat=[
                     dict(
                         loc='x',
                         type='Channel',
                         atte_type='Linear',
                         use_sigmoid=False,
                         use_proj=True,
                         proj_norm=True
                     ),
                     dict(
                         loc='x',
                         type='Spatial',
                         atte_type='Max',
                         use_sigmoid=False,
                         use_proj=True,
                         proj_norm=False
                     ),
                 ],
                 rebuild_loss=[
                     dict(
                         type='L2',
                         gamma=0.01
                     ),
                     dict(
                         type='L2',
                         gamma=0.001
                     ),
                 ],
                 contrast_feat=dict(
                     type='Channel',
                     atte_type='Linear',
                     use_sigmoid=False,
                     use_proj=True,
                     proj_norm=True
                 ),
                 contrast_loss=dict(
                     type='L2',
                     gamma=0.01
                 ),
                 alpha=1,
                 beta=1,
                 fusion_strategy=0,
                 init_cfg=None):
        super(AFF, self).__init__(init_cfg)
        self.planes = planes
        self.ex_planes = ex_planes
        self.alpha = alpha
        self.beta = beta
        r_feat_cfgs = deepcopy(rebuild_feat)
        r_loss_cfgs = deepcopy(rebuild_loss)
        self.rebuilds = nn.ModuleList()
        self.rebuild_src = [feat_cfg.pop('loc')
                            for feat_cfg in r_feat_cfgs]
        self.fusion_strategy = fusion_strategy
        for feat_cfg, loss_cfg in zip(r_feat_cfgs, r_loss_cfgs):
            rebuild_dict = nn.ModuleDict()
            rebuild_dict['feat_org'] = self.build_feat(feat_cfg, planes)
            rebuild_dict['feat_fusion'] = self.build_feat(feat_cfg, planes)
            rebuild_dict['loss'] = self.build_loss(loss_cfg)
            self.rebuilds.append(rebuild_dict)

        self.cf1 = self.build_feat(contrast_feat, planes)
        self.cf2 = self.build_feat(contrast_feat, planes)
        self.loss_c = self.build_loss(contrast_loss)

        self.trans_conv1 = nn.Conv2d(planes, planes,
                                     3, 1, 1, bias=True)
        self.trans_conv2 = nn.Conv2d(planes, planes,
                                     3, 1, 1, bias=True)
        self.trans_conv1.apply(weight_init)
        self.trans_conv2.apply(weight_init)

    def build_feat(self, feat_cfg, planes):
        f_cfg = deepcopy(feat_cfg)
        f_type = f_cfg.pop('type') if f_cfg else None
        if f_type == 'Spatial':
            feat_module = SpatialAttention
            f_out_channel = 1
        elif f_type == 'Channel':
            feat_module = ChannelAttention
            f_out_channel = planes
        elif f_type == 'Feat':
            feat_module = Compress
            f_out_channel = planes // f_cfg['ratio']
        # elif f_type == 'Correlation':
        #     feat_module = Correlation
        #     f_out_channel = planes
        # elif f_type == 'Gram':
        #     feat_module = Gram
        #     f_out_channel = planes
        elif f_type == 'Dummy':
            feat_module = Dummy
            f_out_channel = planes
        else:
            feat_module = Dummy
            f_out_channel = planes
        if f_cfg:
            feat = feat_module(f_out_channel, **f_cfg)
        else:
            feat = feat_module(f_out_channel)
        return feat

    def build_loss(self, loss_cfg):
        l_cfg = deepcopy(loss_cfg)
        assert l_cfg
        l_type = l_cfg.pop('type')

        if l_type == 'L2':
            loss_module = L2_Loss
        elif l_type == 'Cos':
            loss_module = Cos_Loss
        else:
            raise Exception('Loss type: %s undefined' % l_type)
        loss = loss_module(**l_cfg)
        return loss

    def forward(self, x, x_ex, return_loss=False):
        """Forward function."""
        x0 = x
        x_ex0 = x_ex
        loss = x.new_zeros(1)
        org_rfs = []
        for src, rebuild_dict in zip(self.rebuild_src,
                                     self.rebuilds):
            if src == 'x':
                feat = x
            elif src == 'x_ex':
                feat = x_ex
            else:
                raise Exception('Wrong type: %s' % src)
            feat_out = rebuild_dict['feat_org'](feat)
            if 'loss' in feat_out.keys():
                loss = loss + feat_out['loss']
            org_rfs.append(feat_out['atte_map'])

        x = self.trans_conv1(x)
        x_ex = self.trans_conv2(x_ex)
        cf1_out = self.cf1(x)
        cf2_out = self.cf2(x_ex)
        cf1 = cf1_out['atte_map']
        cf2 = cf2_out['atte_map']
        if 'loss' in cf1_out.keys():
            loss = loss + cf1_out['loss']
        if 'loss' in cf2_out.keys():
            loss = loss + cf2_out['loss']
        if self.fusion_strategy == 0:
            x_fusion = x + x_ex + x0
        elif self.fusion_strategy == 1:
            x_fusion = x + x_ex + x0 + x_ex0
        elif self.fusion_strategy == 2:
            x_fusion = x + x_ex
        for src, org_rf, rebuild_dict in zip(self.rebuild_src,
                                             org_rfs,
                                             self.rebuilds):

            feat_out = rebuild_dict['feat_fusion'](x_fusion)
            fusion_rf = feat_out['atte_map']
            if 'loss' in feat_out.keys():
                loss = loss + feat_out['loss']
            loss_rebuild = rebuild_dict['loss'](org_rf, fusion_rf)
            if src == 'x':
                loss = loss + loss_rebuild
            elif src == 'x_ex':
                loss = loss + self.alpha * loss_rebuild
            else:
                raise Exception('Wrong type: %s' % src)

        loss_c = self.loss_c(cf1, cf2)

        loss = loss + self.beta * loss_c
        if return_loss:
            # print(loss)
            return loss, x_fusion
        else:
            return x_fusion
#############################


class CRGs(BaseModule):
    """
    Cfnet: A cross fusion network for joint land cover classification using optical and sar image
    """
    def __init__(self,
                 planes,
                 ex_planes,
                 gate_type='CRGs',
                 trans=True,
                 add_skip=True,
                 init_cfg=None):
        super(CRGs, self).__init__(init_cfg)
        self.planes = planes
        self.ex_planes = ex_planes
        self.gate_type = gate_type
        if gate_type == 'IGs':
            self.gate1 = SEAttention(planes, planes)
            self.gate2 = SEAttention(ex_planes, ex_planes)
        elif gate_type == 'CG1':
            self.gate = SEAttention(planes, planes)
        elif gate_type == 'CG2':
            self.gate = SEAttention(planes + ex_planes, planes)
        if gate_type == 'CRGs':
            self.gate1 = SEAttention(planes, planes)
            self.gate2 = SEAttention(ex_planes, ex_planes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)
        self.trans_conv1 = nn.Conv2d(planes, planes,
                                     3, 1, 1, bias=True)
        self.trans_conv2 = nn.Conv2d(planes, planes,
                                     3, 1, 1, bias=True)
        self.trans_conv1.apply(weight_init)
        self.trans_conv2.apply(weight_init)

    def forward(self, x_input, x_ex_input):
        x = self.trans_conv1(x_input)
        x_ex = self.trans_conv2(x_ex_input)
        if self.gate_type == 'IGs':
            g1 = self.gate1(x)
            g2 = self.gate2(x_ex)
        elif self.gate_type == 'CG1':
            g1 = self.gate(x)
            g2 = 1 - g1
        elif self.gate_type == 'CG2':
            g1 = self.gate(torch.cat([x, x_ex], dim=1))
            g2 = 1 - g1
        elif self.gate_type == 'CRGs':
            g1 = self.gate2(x_ex)
            g2 = self.gate1(x)
        else:
            raise Exception()
        x_fusion = x * g1 + x_ex * g2 + x_input

        return x_fusion


class AFFM(BaseModule):
    """
    Spatial and spectral extraction network with adaptive feature fusion for pansharpening
    https://github.com/RSMagneto/SSE-Net/blob/main/model.py
    """
    def __init__(self,
                 planes,
                 ex_planes,
                 trans=True,
                 add_skip=False,
                 init_cfg=None):
        super(AFFM, self).__init__(init_cfg)
        self.planes = planes
        self.ex_planes = ex_planes
        self.conv = nn.Conv2d(planes*2, planes,
                              kernel_size=3, stride=1, padding=1)
        self.conv_ = nn.Conv2d(planes, planes,
                              kernel_size=3, stride=1, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)
        self.trans = trans
        if self.trans:
            self.trans_conv1 = nn.Conv2d(planes, planes,
                                         3, 1, 1, bias=True)
            self.trans_conv2 = nn.Conv2d(planes, planes,
                                         3, 1, 1, bias=True)
            self.trans_conv1.apply(weight_init)
            self.trans_conv2.apply(weight_init)

    def forward(self, x, x_ex, return_loss=False):
        if self.trans:
            x = self.trans_conv1(x)
            x_ex = self.trans_conv2(x_ex)
        # from https://github.com/RSMagneto/SSE-Net/blob/main/model.py
        image = torch.cat([x, x_ex], 1)#堆叠
        image_ = self.conv(image)
        image_ = self.conv_(image_)#卷积
        mask = F.softmax(image_,dim=1)#缩小到0—1
        out_1 = torch.mul(mask, x)#点乘
        out_2 = torch.mul((1 - mask), x_ex)
        x_f = (out_1 + out_2)

        x_f = x_f + x
        return x_f

class SEAttention(nn.Module):
    def __init__(self, channel, out_channel, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel = channel
        self.out_channel = out_channel
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, out_channel, bias=False),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        gate = self.fc(y).view(b, self.out_channel, 1, 1)
        return gate

class GFF(BaseModule):
    def __init__(self,
                 planes,
                 ex_planes,
                 trans=True,
                 add_skip=True,
                 init_cfg=None):
        super(GFF, self).__init__(init_cfg)
        self.planes = planes
        self.ex_planes = ex_planes
        self.conv_w1 = nn.Conv2d(planes,
                                 1, 1, 1, 0, bias=True)
        self.conv_w2 = nn.Conv2d(planes,
                                 1, 1, 1, 0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.skip = add_skip
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)
        self.trans_conv1 = nn.Conv2d(planes, planes,
                                     3, 1, 1, bias=True)
        self.trans_conv2 = nn.Conv2d(planes, planes,
                                     3, 1, 1, bias=True)
        self.trans_conv1.apply(weight_init)
        self.trans_conv2.apply(weight_init)


    def forward(self, x, x_ex):
        x = self.trans_conv1(x)
        x_ex = self.trans_conv2(x_ex)
        w1 = self.sigmoid(self.conv_w1(x))
        w2 = self.sigmoid(self.conv_w2(x_ex))
        x_f = (1 + w1) * x + (1 - w1) * w2 * x_ex
        if self.skip:
            x_f = x_f + x_ex
        else:
            x_f = x_f#  + x
        return x_f

class GIF(BaseModule):
    """
    Robust deep multi-modal learning based on gated information fusion network
    """
    def __init__(self,
                 planes,
                 ex_planes,
                 trans=True,
                 add_skip=True,
                 init_cfg=None):
        super(GIF, self).__init__(init_cfg)
        self.planes = planes
        self.ex_planes = ex_planes
        self.conv_w1 = nn.Conv2d(2 * planes,
                                 1, 3, 1, 1, bias=True)
        self.conv_w2 = nn.Conv2d(2 * planes,
                                 1, 3, 1, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.conv_j = nn.Conv2d(2 * planes,
                                planes, 1, 1, 0, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)
        self.trans_conv1 = nn.Conv2d(planes, planes,
                                     3, 1, 1, bias=True)
        self.trans_conv2 = nn.Conv2d(planes, planes,
                                     3, 1, 1, bias=True)
        self.trans_conv1.apply(weight_init)
        self.trans_conv2.apply(weight_init)

    def forward(self, x_input, x_ex_input):
        x = self.trans_conv1(x_input)
        x_ex = self.trans_conv2(x_ex_input)

        F_G = torch.cat([x, x_ex], dim=1)
        w1 = self.sigmoid(self.conv_w1(F_G))
        w2 = self.sigmoid(self.conv_w2(F_G))
        F_F = torch.cat([w1 * x, w2 * x_ex], dim=1)
        F_J = torch.relu(self.conv_j(F_F))

        x_f = F_J + x_input
        return x_f


class CRM(BaseModule):
    """
    Calibrated rgb-d salient object detection
    https://github.com/jiwei0921/DCF/blob/main/DCF_code/model/fusion.py

    """
    def __init__(self,
                 planes,
                 ex_planes,
                 trans=True,
                 add_skip=True,
                 init_cfg=None):
        super(CRM, self).__init__(init_cfg)
        self.planes = planes
        self.ex_planes = ex_planes
        self.squeeze_x = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_x = nn.Sequential(
            nn.Conv2d(planes, planes, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid())


        self.squeeze_ex = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_ex = nn.Sequential(
            nn.Conv2d(planes, planes, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid())

        self.cross_conv = nn.Conv2d(planes*2, planes,
                                    1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

        self.trans_conv1 = nn.Conv2d(planes, planes,
                                     3, 1, 1, bias=True)
        self.trans_conv2 = nn.Conv2d(planes, planes,
                                     3, 1, 1, bias=True)
        self.trans_conv1.apply(weight_init)
        self.trans_conv2.apply(weight_init)

    def forward(self, x_input, x_ex_input):
        x = self.trans_conv1(x_input)
        x_ex = self.trans_conv2(x_ex_input)
        SCA_x_ca = self.channel_attention_x(self.squeeze_x(x))
        SCA_x_o = x * SCA_x_ca.expand_as(x)

        SCA_ex_ca = self.channel_attention_ex(self.squeeze_ex(x_ex))
        SCA_ex_o = x_ex * SCA_ex_ca.expand_as(x_ex)

        Co_ca = torch.softmax(SCA_x_ca + SCA_ex_ca, dim=1)
        SCA_x_co = x * Co_ca.expand_as(x)
        SCA_ex_co = x_ex * Co_ca.expand_as(x_ex)

        CR_fea_x = SCA_x_o + SCA_x_co
        CR_fea_ex = SCA_ex_o + SCA_ex_co

        CR_fea = torch.cat([CR_fea_x, CR_fea_ex],
                           dim=1)
        CR_fea = self.cross_conv(CR_fea)

        x_f = CR_fea + x_input
        return x_f

class CWF(BaseModule):
    """
    Abmdrnet: Adaptive-weighted bi-directional modality difference reduction network for rgb-t semantic segmentation
    """
    def __init__(self,
                 planes,
                 ex_planes,
                 add_skip=True,
                 trans=True,
                 init_cfg=None):
        super(CWF, self).__init__(init_cfg)
        self.planes = planes
        self.ex_planes = ex_planes
        self.Conv = nn.Sequential(
            nn.Conv2d(2 * planes,
                      planes, 1, 1, 0, bias=True),
            nn.Conv2d(planes,
                      planes, 3, 1, 1, bias=True)
        )
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.Conv.apply(weight_init)
        self.trans_conv1 = nn.Conv2d(planes, planes,
                                     3, 1, 1, bias=True)
        self.trans_conv2 = nn.Conv2d(planes, planes,
                                     3, 1, 1, bias=True)
        self.trans_conv1.apply(weight_init)
        self.trans_conv2.apply(weight_init)



    def forward(self, x_input, x_ex_input):
        x = self.trans_conv1(x_input)
        x_ex = self.trans_conv2(x_ex_input)
        w_x = torch.cat([x, x_ex], dim=1)
        W = self.GAP(self.sigmoid(self.Conv(w_x)))

        x_f = W * x + (1 - W) * x_ex + x_input
        return x_f

class ADD(BaseModule):

    def __init__(self,
                 planes,
                 ex_planes,
                 add_skip=False,
                 trans=False,
                 init_cfg=None):
        super(ADD, self).__init__(init_cfg)
        self.trans_conv1 = nn.Conv2d(planes, planes,
                                     3, 1, 1, bias=True)
        self.trans_conv2 = nn.Conv2d(planes, planes,
                                     3, 1, 1, bias=True)
        self.trans_conv1.apply(weight_init)
        self.trans_conv2.apply(weight_init)



    def forward(self, x, x_ex):
        x_f = self.trans_conv1(x) + self.trans_conv2(x_ex) + x
        return x_f


@NECKS.register_module()
class DualFusionFPN(BaseModule):
    # General two stream fusion neck

    def __init__(self,
                 neck_cfg,
                 fusion_cfg,
                 num_levels=5,
                 init_cfg=None):

        super().__init__(init_cfg)
        self.neck = NECKS.build(neck_cfg)
        self.neck_ex = NECKS.build(neck_cfg)
        ############
        self.fusion_cfg = deepcopy(fusion_cfg)
        self.fusion_type = fusion_cfg.pop('type')
        self.fusion_settings = fusion_cfg
        print(self.fusion_settings)
        ###################################################################
        self.fusion_convs = nn.ModuleList()
        """
        """
        if self.fusion_type == 'GIF':
            fusion_module = GIF
        elif self.fusion_type == 'CRM':
            fusion_module = CRM
        elif self.fusion_type == 'CWF':
            fusion_module = CWF
        elif self.fusion_type == 'CRGs':
            fusion_module = CRGs
        elif self.fusion_type == 'AFFM':
            fusion_module = AFFM
        elif self.fusion_type == 'ADD':
            fusion_module = ADD
        elif self.fusion_type == 'AFF':
            fusion_module = AFF
        else:
            raise Exception('Fusion type undefined: %s'
                            % self.fusion_type)
        out_channels = self.neck.out_channels
        for i in range(num_levels):
            fusion_block = fusion_module(out_channels,
                                         out_channels,
                                         **self.fusion_settings)
            self.fusion_convs.append(fusion_block)

    def forward(self, inputs, return_loss=False):
        x, x_ex = inputs
        outs = self.neck(x)
        outs_ex = self.neck_ex(x_ex)
        # fusion before output
        fusion_outs = []
        losses = []
        for l, o1, o2 in zip(range(len(outs)), outs, outs_ex):
            if self.fusion_type == 'AFF':
                if return_loss:
                    loss, fusion_out = self.fusion_convs[l](o1, o2,
                                                            return_loss=True)
                    losses.append(loss)
                else:
                    fusion_out = self.fusion_convs[l](o1, o2,
                                                      return_loss=False)
            else:
                fusion_out = self.fusion_convs[l](o1, o2)
            fusion_outs.append(fusion_out)
        if return_loss:
            l = outs[0].new_zeros(1)
            for i in range(0, len(losses)):
                l = l + losses[i]
            if len(losses) > 0:
                losses = l / len(losses)
            else:
                losses = l
            return tuple(fusion_outs), losses
        else:
            return tuple(fusion_outs)
