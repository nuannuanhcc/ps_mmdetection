# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
import numpy as np


def circle_loss(
    sim_ap: torch.Tensor,
    sim_an: torch.Tensor,
    scale: float = 16.0,
    margin: float = 0.1,
    redection: str = "mean"
):
    pair_ap = -scale * (sim_ap - margin)
    pair_an = scale * sim_an
    pair_ap = torch.logsumexp(pair_ap, dim=1)
    pair_an = torch.logsumexp(pair_an, dim=1)
    loss = torch.nn.functional.softplus(pair_ap + pair_an)
    if redection == "mean":
        loss = loss.mean()
    elif redection == "sum":
        loss = loss.sum()
    return loss


@torch.no_grad()
def update_queue(queue, pointer, new_item):
    n = new_item.shape[0]
    length = queue.shape[0]
    if pointer + n <= length:
        queue[pointer: pointer + n] = new_item
        pointer = pointer + n
    else:
        res = n-(length-pointer)
        queue[pointer: length] = new_item[:length-pointer]
        queue[: res] = new_item[-res:]
        pointer = res
    return queue, pointer


class OIM(Function):
    @staticmethod
    def forward(ctx, inputs, targets, lut, queue, num_gt, momentum):
        ctx.lut = lut
        ctx.queue = queue
        ctx.num_gt = num_gt
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs_labeled = inputs.mm(ctx.lut.t())
        outputs_unlabeled = inputs.mm(ctx.queue.t())
        return torch.cat((outputs_labeled, outputs_unlabeled), 1)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_outputs, = grad_outputs
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(torch.cat((ctx.lut, ctx.queue), 0))

        for i, (x, y) in enumerate(zip(inputs, targets)):
            if y == -1:
                tmp = torch.cat((ctx.queue[1:], x.view(1, -1)), 0)
                ctx.queue[:, :] = tmp[:, :]
            elif 0 <= y < len(ctx.lut):
                if i < ctx.num_gt:
                    ctx.lut[y] = ctx.momentum * ctx.lut[y] + (1. - ctx.momentum) * x
                    ctx.lut[y] = F.normalize(ctx.lut[y], dim=-1)
            else:
                continue
        return grad_inputs, None, None, None, None, None


class OIMLossComputation(nn.Module):
    def __init__(self, cfg):
        super(OIMLossComputation, self).__init__()
        self.cfg = cfg
        if self.cfg.dataset_type == 'SysuDataset':
            self.num_pid = 5532
            self.queue_size = 5000
        elif self.cfg.dataset_type == 'PrwDataset':
            self.num_pid = 483
            self.queue_size = 500
        else:
            raise KeyError(cfg.DATASETS.TRAIN)

        self.lut_momentum = 0.0
        self.out_channels = 2048

        self.register_buffer('lut', torch.zeros(self.num_pid, self.out_channels).cuda())
        self.register_buffer('queue', torch.zeros(self.queue_size, self.out_channels).cuda())

    def forward(self, features, gt_labels):

        pids = torch.cat([i[:, -1] for i in gt_labels])
        num_gt = pids.shape[0]

        reid_result = OIM.apply(features, pids, self.lut, self.queue, num_gt, self.lut_momentum)
        loss_weight = torch.cat([torch.ones(self.num_pid).cuda(), torch.zeros(self.queue_size).cuda()])

        scalar = 10
        loss_reid = F.cross_entropy(reid_result * scalar, pids, weight=loss_weight, ignore_index=-1)
        return loss_reid


class CIRCLELossComputation(nn.Module):
    def __init__(self, cfg):
        super(CIRCLELossComputation, self).__init__()
        self.cfg = cfg

        if self.cfg.dataset_type == 'SysuDataset':
            num_labeled = 8192
            num_unlabeled = 8192
        elif self.cfg.dataset_type == 'Prwdataset_type':
            num_labeled = 8192
            num_unlabeled = 8192
        else:
            raise KeyError(cfg.DATASETS.TRAIN)

        self.out_channels = 2048

        self.register_buffer('pointer', torch.zeros(2, dtype=torch.int).cuda())
        self.register_buffer('id_inx', -torch.ones(num_labeled, dtype=torch.long).cuda())
        self.register_buffer('lut', torch.zeros(num_labeled, self.out_channels).cuda())
        self.register_buffer('queue', torch.zeros(num_unlabeled, self.out_channels).cuda())

    def forward(self, features, gt_labels):

        pids = torch.cat([i[:, -1] for i in gt_labels])
        id_labeled = pids[pids > -1]
        feat_labeled = features[pids > -1]
        feat_unlabeled = features[pids == -1]

        if not id_labeled.numel():
            loss = F.cross_entropy(features.mm(self.lut.t()), pids, ignore_index=-1)
            return loss

        self.lut, _ = update_queue(self.lut, self.pointer[0], feat_labeled)

        self.id_inx, self.pointer[0] = update_queue(self.id_inx, self.pointer[0], id_labeled)
        self.queue, self.pointer[1] = update_queue(self.queue, self.pointer[1], feat_unlabeled)

        queue_sim = torch.mm(feat_labeled, self.queue.t())
        lut_sim = torch.mm(feat_labeled, self.lut.t())
        positive_mask = id_labeled.view(-1, 1) == self.id_inx.view(1, -1)
        sim_ap = lut_sim.masked_fill(~positive_mask, float("inf"))
        sim_an = lut_sim.masked_fill(positive_mask, float("-inf"))
        sim_an = torch.cat((queue_sim, sim_an), dim=-1)

        pair_loss = circle_loss(sim_ap, sim_an)
        return pair_loss


class OIMLossComputation_UN(nn.Module):
    def __init__(self, cfg):
        super(OIMLossComputation_UN, self).__init__()
        self.cfg = cfg
        if self.cfg.dataset_type == 'SysuDataset':
            self.num_pid = 15080  # 15080/55260
        elif self.cfg.dataset_type == 'PrwDataset':
            self.num_pid = 483
        else:
            raise KeyError(cfg.DATASETS.TRAIN)

        self.m = 0.0
        self.out_channels = 2048

        self.register_buffer('lut', torch.zeros(self.num_pid, self.out_channels).cuda())

    def forward(self, features, gt_labels):

        pids = torch.cat([i[:, -1] for i in gt_labels])
        id_labeled = pids[pids > -1]
        feat_labeled = features[pids > -1]

        if not id_labeled.numel():
            loss = F.cross_entropy(features.mm(self.lut.t()), pids, ignore_index=-1)
            return loss

        sim_all = HM.apply(feat_labeled, id_labeled, self.lut, self.m)

        scalar = 10
        loss = F.cross_entropy(sim_all * scalar, id_labeled)
        return loss


class CIRCLELossComputation_UN(nn.Module):
    def __init__(self, cfg):
        super(CIRCLELossComputation_UN, self).__init__()
        self.cfg = cfg

        if self.cfg.dataset_type == 'SysuDataset':
            num_labeled = 15080  # 15080/55260
        elif self.cfg.dataset_type == 'Prwdataset_type':
            num_labeled = 8192
        else:
            raise KeyError(cfg.DATASETS.TRAIN)

        self.out_channels = 2048

        self.register_buffer('pointer', torch.zeros(1, dtype=torch.int).cuda())
        self.register_buffer('id_inx', -torch.ones(num_labeled, dtype=torch.long).cuda())
        self.register_buffer('lut', torch.zeros(num_labeled, self.out_channels).cuda())

    def forward(self, features, gt_labels):

        pids = torch.cat([i[:, -1] for i in gt_labels])
        id_labeled = pids[pids > -1]
        feat_labeled = features[pids > -1]
        feat_unlabeled = features[pids == -1]

        if not id_labeled.numel():
            loss = F.cross_entropy(features.mm(self.lut.t()), pids, ignore_index=-1)
            return loss

        self.lut, _ = update_queue(self.lut, self.pointer, feat_labeled)
        self.id_inx, self.pointer = update_queue(self.id_inx, self.pointer, id_labeled)

        lut_sim = torch.mm(feat_labeled, self.lut.t())
        positive_mask = id_labeled.view(-1, 1) == self.id_inx.view(1, -1)
        sim_ap = lut_sim.masked_fill(~positive_mask, float("inf"))
        sim_an = lut_sim.masked_fill(positive_mask, float("-inf"))

        pair_loss = circle_loss(sim_ap, sim_an)
        return pair_loss


class HM(Function):
    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)
        # momentum update
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()
        return grad_inputs, None, None, None

class HybridMemory(nn.Module):
    def __init__(self, cfg, use_circle_loss=False):
        super(HybridMemory, self).__init__()
        self.cfg = cfg
        self.use_circle_loss = use_circle_loss

        if self.cfg.dataset_type == 'SysuDataset':
            num_labeled = 15080  # 15080/55260
            num_unlabeled = 8192
        elif self.cfg.dataset_type == 'PrwDataset':
            num_labeled = 8192
            num_unlabeled = 8192
        else:
            raise KeyError(cfg.DATASETS.TRAIN)

        self.m = 0.2
        self.temp = 0.05
        self.out_channels = 2048

        self.register_buffer('labels', torch.arange(num_labeled, dtype=torch.long).cuda())
        self.register_buffer('features', torch.zeros(num_labeled, self.out_channels).cuda())

    def forward(self, features, gt_labels):
        pids = torch.cat([i[:, -1] for i in gt_labels])
        id_labeled = pids[pids > -1]
        feat_labeled = features[pids > -1]

        if not id_labeled.numel():
            loss = F.cross_entropy(features.mm(self.features.t()), pids, ignore_index=-1)
            return loss

        sim_all = HM.apply(feat_labeled, id_labeled, self.features, self.m)
        targets = self.labels[id_labeled]

        if self.use_circle_loss:
            positive_mask = targets.view(-1, 1) == self.labels.view(1, -1)
            sim_ap = sim_all.masked_fill(~positive_mask, float("inf"))
            sim_an = sim_all.masked_fill(positive_mask, float("-inf"))
            loss = circle_loss(sim_ap, sim_an)
            return loss

        sim_all /= self.temp
        N = sim_all.shape[0]

        labels = self.labels.clone()

        sim = torch.zeros(labels.max()+1, N).float().cuda()
        sim.index_add_(0, labels, sim_all.t().contiguous())
        nums = torch.zeros(labels.max()+1, 1).float().cuda()
        nums.index_add_(0, labels, torch.ones(labels.shape[0], 1).float().cuda())

        sim = sim / nums.expand_as(sim)
        loss = F.cross_entropy(sim.t(), targets)

        return loss


def make_reid_loss_evaluator(cfg):
    # loss_evaluator = OIMLossComputation(cfg)
    # loss_evaluator = OIMLossComputation_UN(cfg)
    # loss_evaluator = CIRCLELossComputation(cfg)
    # loss_evaluator = CIRCLELossComputation_UN(cfg)
    loss_evaluator = HybridMemory(cfg, use_circle_loss=True)
    return loss_evaluator
