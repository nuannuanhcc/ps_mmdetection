# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings
import tqdm
import torch

import mmcv
from .base_runner import BaseRunner
from .builder import RUNNERS
from .checkpoint import save_checkpoint
from .utils import get_host_info
import numpy as np
import collections


@RUNNERS.register_module()
class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def extract_feats(self, data_loader):
        # train_loader+test_forward
        from mmcv.runner import get_dist_info
        rank, world_size = get_dist_info()
        dataset = data_loader.dataset
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(dataset))
        self.model.eval()
        self.mode = 'val'
        features = []
        pids = []
        imgids = []
        print('features extracting ')
        for i, data_batch in enumerate(data_loader):
            if rank == 0:
                batch_size = 1
                for _ in range(batch_size * world_size):
                    prog_bar.update()
            data = data_batch.copy()
            with_unlabeled = True
            from mmcv.parallel import DataContainer as DC
            pid = data['gt_labels']._data[0][0][:, 1]
            imgid = data['gt_labels']._data[0][0][:, 2]
            if not with_unlabeled:
                idx = pid > -1
                pid_labeled = pid[idx]
                imgid_labeled = imgid[idx]
                if pid_labeled.numel() == 0:
                    pids.append(None)
                    imgids.append(None)
                    features.append(None)
                    continue
                pids.append(pid_labeled)
                imgids.append(imgid_labeled)
                gt_bboxes = [[data['gt_bboxes']._data[0][0][idx]]]
            else:
                pids.append(pid)
                imgids.append(imgid)
                gt_bboxes = [[data['gt_bboxes']._data[0][0]]]

            data = dict(
                img=[data['img']._data[0]],
                img_metas=[data['img_metas']],
                gt_bboxes=gt_bboxes
            )
            with torch.no_grad():
                img = data['img'][0].clone()
                box = data['gt_bboxes'].copy()[0][0]
                box = torch.round(box).int().tolist()
                img_crop = [torch.nn.functional.interpolate(img[:, :, b[1]:b[3], b[0]:b[2]], size=(7 * 16, 7 * 16)) for b in box]
                img_crop = torch.cat(img_crop).cuda()
                data['img_metas'][0]._data[0][0]['img_crop'] = img_crop

                _, feats = self.model(return_loss=False, **data)
                features.append(feats)
        if world_size > 1:
            from ..test import collect_results_cpu
            features = collect_results_cpu(features, len(dataset))
            pids = collect_results_cpu(pids, len(dataset))
            imgids = collect_results_cpu(imgids, len(dataset))
        if rank == 0:
            features = torch.cat([torch.from_numpy(i) for i in features if i is not None])
            self.pids = torch.cat([i.unsqueeze(-1) for i in pids if i is not None]).squeeze()
            self.imgids = torch.cat([i.unsqueeze(-1) for i in imgids if i is not None]).squeeze()
            self.model.module.reid_head.loss_evaluator.features = torch.nn.functional.normalize(features, dim=1).cuda()
            del data_loader, features

    def conduct_cluster(self):
        self.logger.info('Start clustering')
        start_time = time.time()
        features = self.model.module.reid_head.loss_evaluator.features.clone()
        sim = torch.mm(features, features.t())
        del features

        # cluster1
        neb = 2
        sim_k, idx = torch.topk(sim, neb, dim=-1)
        label = torch.arange(idx.shape[0])
        for i in idx:
            if self.imgids[i[0]] == self.imgids[i[1]]:
                continue
            # min_idx = torch.min(i)
            # min_val = label[min_idx].clone()
            min_val = torch.min(label[i])
            for j in range(neb):
                label[i[j]] = min_val

        # cluster3
        # n = sim.shape[0]
        # label = torch.arange(n)
        # sim = sim.mul(1 - torch.eye(n).cuda())
        # while sim.max() > 0.5:
        #     max_val = torch.max(sim)
        #     max_idx = torch.argmax(sim)
        #     x, y = max_idx // n, max_idx % n
        #     if self.imgids[x] == self.imgids[y]:
        #         sim[x, y] = 0
        #     else:
        #         label[x] = label[y]
        #         sim[x, :] = 0
        #         sim[:, y] = 0

        label_set = set(label.tolist())
        map_label = {label: new for new, label in enumerate(label_set)}
        pseudo_labels = np.array([map_label[i.item()] for i in label])

        num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
        total_time = time.time() - start_time
        self.logger.info('End clustering, total time: %3f', total_time)
        # generate new dataset and calculate cluster centers
        labels = []
        outliers = 0
        for id in pseudo_labels:
            if id != -1:
                labels.append(id)
            else:
                labels.append(num_ids + outliers)
                outliers += 1
        labels = torch.Tensor(labels).long().cuda()
        self.model.module.reid_head.loss_evaluator.labels = labels

        from sklearn import metrics
        true_labels = self.pids.cpu().numpy()
        pred_labels = labels.cpu().numpy()
        cluster_metric = metrics.adjusted_rand_score(true_labels, pred_labels)
        self.logger.info('cluster_metric is %f', cluster_metric)

        # statistics of clusters and un-clustered instances
        index2label = collections.defaultdict(int)
        for label in labels:
            index2label[label.item()] += 1
        index2label = np.fromiter(index2label.values(), dtype=float)
        self.logger.info('Statistics for epoch %d: %d clusters, %d un-clustered instances',
                         self._epoch, (index2label > 1).sum(), (index2label == 1).sum())

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            with torch.no_grad():
                self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, cluster_loader,  data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')
        self.extract_feats(cluster_loader)

        while self.epoch < self._max_epochs:
            from mmcv.runner import get_dist_info
            rank, world_size = get_dist_info()
            if rank == 0:
                self.conduct_cluster()
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)


@RUNNERS.register_module()
class Runner(EpochBasedRunner):
    """Deprecated name of EpochBasedRunner."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            'Runner was deprecated, please use EpochBasedRunner instead')
        super().__init__(*args, **kwargs)
