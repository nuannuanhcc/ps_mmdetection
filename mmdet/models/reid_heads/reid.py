import torch
import torch.nn.functional as F
from torch import nn
from .loss import make_reid_loss_evaluator


class REIDModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and outputs
    RPN proposals and losses. Works for both FPN and non-FPN.
    """
    def __init__(self, cfg):
        super(REIDModule, self).__init__()
        self.cfg = cfg
        self.loss_evaluator = make_reid_loss_evaluator(cfg)
        self.fc = nn.Linear(256 * 7 * 7, 2048)

    def forward(self, x, x_crop=None, gt_labels=None):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        feats = F.normalize(x, dim=-1)

        if x_crop is not None:
            x_crop = x_crop.view(x_crop.size(0), -1)
            x_crop = self.fc(x_crop)
            feats_crop = F.normalize(x_crop, dim=-1)

        if not self.training:
            return feats
        loss_reid = self.loss_evaluator(feats_crop, feats, gt_labels)
        return {"loss_reid": [loss_reid], }


def build_reid(cfg):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    return REIDModule(cfg)
