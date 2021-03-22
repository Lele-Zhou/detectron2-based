import numpy as np
import torch
import torch.nn as nn


from detectron2.config import CfgNode as CN
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry


VERTEX_HEAD_REGISTRY = Registry("VERTEX_HEAD")
VERTEX_HEAD_REGISTRY.__doc__ = """
Registry for point head, which makes prediction for a given set of plane features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def vertex_loss(vertex_logits, vertex_gt):
    """
    预测输出0-1之间的数字直接与 junction_heatmap 做loss 存在的问题
    1.如果预测偏一个像素与偏几个像素的loss是相同的，而我们希望偏一个像素的loss很小，偏很多像素的loss要大
    即loss应该也与点的位置相关。
    """
    with torch.no_grad():
        vertex_loss = nn.BCELoss(vertex_gt, vertex_logits)
    return vertex_loss


@VERTEX_HEAD_REGISTRY.register()
class VertexHead(nn.Module):
    def __int__(self, cfg, input_shape: ShapeSpec):
        super(VertexHead, self).__int__()

        self.input_channels = input_shape.channels
        # conv_dim = cfg.MODEL.VERTEX_HEAD.CONV_DIM
        self.output_channels = cfg.MODEL.VERTEX_HEAD.OUTPUT_CHANNELS
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=self.output_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_map):
        out = self.conv1(feature_map)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn(out)
        out = self.relu(out)

        return out


def build_vertex_head(cfg, input_channels):
    """
    Build a vertex head defined by `cfg.MODEL.VERTEX_HEAD.NAME`.
    """
    head_name = cfg.MODEL.VERTEX_HEAD.NAME
    return VERTEX_HEAD_REGISTRY.get(head_name)(cfg, input_channels)