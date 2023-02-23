import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES, build_backbone
from ..utils import ResLayer


@BACKBONES.register_module()
class TFResNet(BaseModule):

    def __init__(self,
                 backbone,
                 backbone_extra,
                 backbone_channel=1,
                 backbone_extra_channel=4,
                 pretrained=None,
                 fusion_type=0,
                 init_cfg=None):
        super(TFResNet, self).__init__(init_cfg)
        backbone.pretrained = pretrained
        backbone_extra.pretrained = pretrained
        self.fusion_type = fusion_type
        self.backbone = build_backbone(backbone)
        self.backbone_extra = build_backbone(backbone_extra)

        self.BEC = backbone_extra_channel
        self.BC = backbone_channel


    def forward(self, img):
        img_pan = img[:, :self.BC, :, :]
        img_extra = img[:, self.BC:, :, :]
        x = self.backbone(img_pan)
        if getattr(self, 'backbone_extra', None):
            x_extra = self.backbone_extra(img_extra)
        else:
            x_extra = self.backbone(img_extra, extra=True)
        if self.fusion_type == 0:
            return (x, x_extra)
        elif self.fusion_type == 1:
            pan_weight = 1
            extra_weight = 1
            if self.weight:
                pan_weight = self.weight
                extra_weight = 1 - pan_weight
            assert pan_weight >= 0 and pan_weight <= 1
            assert extra_weight >= 0 and extra_weight <= 1
            x = [x_ * pan_weight + x_extra_ * extra_weight for x_, x_extra_ in zip(x, x_extra)]
            return x
        else:
            raise NotImplementedError
