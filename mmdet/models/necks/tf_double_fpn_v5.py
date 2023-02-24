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
                self.w = nn.Parameter(torch.ones(1))
                self.b = nn.Parameter(torch.rand(1))

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
                atte_map = self.w * atte_map + self.b
        elif self.atte_type == 'Mean':
            mean_out = torch.mean(x, dim=1, keepdim=True)
            atte_map = mean_out
            if self.use_proj:
                atte_map = self.w * atte_map + self.b
        if self.use_sigmoid:
            atte_map = self.sigmoid(atte_map)
        outs['atte_map'] = atte_map
        if self.proj_norm and self.use_proj:
            outs['loss'] = torch.abs(self.w - x.new_ones(1))
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
                init_U = torch.randn(in_planes, in_planes) * 0.1
                init_U = init_U / torch.norm(init_U, dim=1, keepdim=True)
                self.U = nn.Parameter(init_U)
                self.bias = nn.Parameter(torch.randn(in_planes))
        elif self.atte_type == 'Max':
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        outs = dict()
        if self.atte_type == 'V1':
            atte_map = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        elif self.atte_type == 'Linear':
            if self.use_proj:
                x = x.reshape(B, C, H * W)
                U = self.U[None, ...].expand(B, C, C)
                b = self.bias[None, ...].expand(B, C)[..., None]
                atte_map = torch.matmul(U, x) + b
                atte_map = atte_map.reshape(B, C, H, W)
                if self.proj_norm:
                    utu = torch.matmul(self.U, self.U.transpose(1, 0))
                    target = torch.eye(len(self.U), device=x.device)
                    F2norm = torch.dist(utu, target, p=2)
                    outs['loss'] = F2norm
            else:
                atte_map = x
            atte_map = self.max_pool(atte_map)

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


class RebuildFusion(BaseModule):

    def __init__(self,
                 planes,
                 ex_planes,
                 rebuild_feat=[
                     dict(
                         loc='x',
                         type='Spatial'
                     ),
                     dict(
                         loc='x_ex',
                         type='Channel'
                     )
                 ],
                 rebuild_loss=[
                     dict(
                         loc='x',
                         type='L2'
                     ),
                     dict(
                         loc='x_ex',
                         type='L2'
                     ),
                 ],
                 contrast_feat=dict(
                     type='Channel'
                 ),
                 contrast_loss=dict(
                     type='L2'
                 ),
                 fusion_strategy=0,
                 alpha=1,
                 beta=1,
                 init_cfg=None):
        super(RebuildFusion, self).__init__(init_cfg)
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
        elif f_type == 'Correlation':
            feat_module = Correlation
            f_out_channel = planes
        elif f_type == 'Gram':
            feat_module = Gram
            f_out_channel = planes
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


@NECKS.register_module()
class TFDoubleFPNv5(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:

        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 fusion_cfg=dict(
                     type='Simple',
                     fusion_type='Add'
                 ),
                 conv_weight_norm=False,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        # fusion_extra_convs: list, contain conv kernel size
        print(conv_weight_norm)
        super(TFDoubleFPNv5, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        ###################################################################
        self.conv_weight_norm = conv_weight_norm
        self.lateral1_convs = self.build_laterals()
        self.lateral2_convs = self.build_laterals()
        self.fpn1_convs = self.build_fpn()
        self.fpn2_convs = self.build_fpn()

        self.fusion_cfg = deepcopy(fusion_cfg)
        self.fusion_type = fusion_cfg.pop('type')
        self.fusion_settings = fusion_cfg
        print(self.fusion_settings)

        ###################################################################
        self.fusion_convs = nn.ModuleList()
        if self.fusion_type == 'Rebuild':
            fusion_module = RebuildFusion
        else:
            raise Exception('Fusion type undefined: %s'
                            % self.fusion_type)
        ###################################################################
        for i in range(self.start_level, self.backbone_end_level):
            fusion_block = fusion_module(out_channels,
                                         out_channels,
                                         **self.fusion_settings)
            self.fusion_convs.append(fusion_block)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if extra_levels >= 1:
            for i in range(extra_levels):
                fusion_block = fusion_module(out_channels,
                                             out_channels,
                                             **self.fusion_settings)
                self.fusion_convs.append(fusion_block)
        print(self)
        # for p in self.parameters():
        #     print(p.shape)

    @auto_fp16()
    def forward_single_stream(self, inputs, lateral_convs, fpn_convs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(fpn_convs[i](outs[-1]))
        return tuple(outs)

    @auto_fp16()
    def forward(self, inputs, return_loss=False):
        """Forward function."""
        x1, x2 = inputs
        outs1 = self.forward_single_stream(x1, self.lateral1_convs,
                                           self.fpn1_convs)
        outs2 = self.forward_single_stream(x2, self.lateral2_convs,
                                           self.fpn2_convs)
        fusion_outs = []
        losses = []
        for l, o1, o2 in zip(range(len(outs1)), outs1, outs2):
            if return_loss:
                loss, fusion_out = self.fusion_convs[l](o1, o2,
                                                        return_loss=True)
                losses.append(loss)
            else:
                fusion_out = self.fusion_convs[l](o1, o2,
                                                  return_loss=False)
            fusion_outs.append(fusion_out)
        if return_loss:
            l = outs1[0].new_zeros(1)
            for i in range(0, len(losses)):
                l = l + losses[i]
            losses = l / len(losses)
            return tuple(fusion_outs), losses
        else:
            return tuple(fusion_outs)

    ###########################################################################################
    def build_laterals(self):
        in_channels = self.in_channels
        out_channels = self.out_channels
        lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=self.act_cfg,
                inplace=False)
            if self.conv_weight_norm:
                l_conv.conv = nn.utils.weight_norm(l_conv.conv)
            lateral_convs.append(l_conv)
        return lateral_convs

    def build_fpn(self):

        fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            fpn_conv = ConvModule(
                self.out_channels,
                self.out_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            if self.conv_weight_norm:
                fpn_conv.conv = nn.utils.weight_norm(fpn_conv.conv)
            fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = self.num_outs - self.backbone_end_level \
                       + self.start_level

        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = self.out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    self.out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False)
                if self.conv_weight_norm:
                    extra_fpn_conv.conv = nn.utils.weight_norm(extra_fpn_conv.conv)
                fpn_convs.append(extra_fpn_conv)
        return fpn_convs
