import torch
import torch.nn as nn

from mmdet.core import bbox2result, bbox2roi
from ..builder import DETECTORS, build_backbone, build_head, build_neck, build_roi_extractor
from .base import BaseDetector
from mmdet.models.reid_heads.reid import build_reid


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        if test_cfg.with_reid:
            self.reid_head = build_reid(test_cfg)
            self.bbox_roi_extractor = build_roi_extractor(test_cfg.roi_extractor)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()
        if self.test_cfg.with_reid:
            self.bbox_roi_extractor.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        x = self.extract_feat(img)
        cls_labels = [i[:, 0] for i in gt_labels] if self.train_cfg.with_reid else gt_labels
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              cls_labels, gt_bboxes_ignore)
        if self.train_cfg.with_reid:
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], bbox2roi(gt_bboxes))
            loss_reid = self.reid_head(bbox_feats, gt_labels)
            losses.update(loss_reid)
        return losses

    def simple_test(self, img, img_metas, rescale=False, gt_bboxes=None):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        # backbone
        x = self.extract_feat(img)
        # person search -- query
        if gt_bboxes is not None:
            gt_bbox_list = gt_bboxes[0][0]  # [n, 4]
            gt_bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], bbox2roi([gt_bbox_list]))
            gt_bbox_feats = self.reid_head(gt_bbox_feats)
            gt_bbox_list = torch.cat([gt_bbox_list / img_metas[0]['scale_factor'][0],  # TODO multi-scale
                                      torch.ones(gt_bbox_list.shape[0], 1).cuda()], dim=-1)
            bbox_results = [bbox2result(gt_bbox_list, torch.zeros(gt_bbox_list.shape[0]), self.bbox_head.num_classes)]
            return bbox_results, gt_bbox_feats.cpu().numpy()
        # boxes detection
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list
        # only detection
        if not self.test_cfg.with_reid:
            return bbox_results[0]
        # person search -- gallery
        pre_bbox_list = bbox_list[0][0] * img_metas[0]['scale_factor'][0]
        pre_bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], bbox2roi([pre_bbox_list]))
        pre_bbox_feats = self.reid_head(pre_bbox_feats)
        return bbox_results, pre_bbox_feats.cpu().numpy()

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        return [self.bbox_head.aug_test(feats, img_metas, rescale=rescale)]
