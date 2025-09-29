

import torch

from mmtrack.models.builder import MODELS
from mmtrack.models.mot import OCSORT

from core.utils import results2outs, outs2results

import pprint

import numpy as np


@MODELS.register_module()
class PseudoUncertainMOT(OCSORT):
    """
    """

    def __init__(self,
                 detector=None,
                 tracker=None,
                 motion=None,
                 init_cfg=None):
        super().__init__(detector, tracker, motion, init_cfg)
        self.fixed_cov_coeff = 1.0
        pprint.pprint(f"Initialized PseudoUncertainMOT with fixed covariance coeff {self.fixed_cov_coeff}")

    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        """Test without augmentations.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool, optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to False.

        Returns:
            dict[str : list(ndarray)]: The tracking results.
        """
        frame_id = img_metas[0].get('frame_id', -1)
        if frame_id == 0:
            self.tracker.reset()

        det_results = self.detector.simple_test(
            img, img_metas, rescale=rescale)
        assert len(det_results) == 1, 'Batch inference is not supported.'
        print("Detection results[0] shape:", np.array(det_results[0]).shape)
        # bbox_results = det_results[0]['bbox']
        bbox_results = det_results[0]
        # bbox_cov_results = det_results[0]['bbox_cov']s  --- IGNORE ---
        num_classes = len(bbox_results)
        if hasattr(self.detector, 'bbox_head'):
            assert num_classes == self.detector.bbox_head.num_classes, \
                'Number of classes mismatch.'
        else:
            raise TypeError('detector must have bbox head.')

        bbox_cov_results = [
            self.fixed_cov_coeff * np.eye(4, dtype=np.float32)[None, ...].repeat(len(bbox_results[i]), axis=0)
            for i in range(num_classes)
        ]

        outs_det = results2outs(bbox_results=bbox_results, bbox_cov_results=bbox_cov_results)
        det_bboxes = torch.from_numpy(outs_det['bboxes']).to(img)
        # det_bbox_covs = torch.from_numpy(outs_det['bbox_covs']).to(img) --- IGNORE ---
        det_labels = torch.from_numpy(outs_det['labels']).to(img).long()

        track_bboxes, track_bbox_covs, track_labels, track_ids = self.tracker.track(
            img=img,
            img_metas=img_metas,
            model=self,
            bboxes=det_bboxes,
            # bbox_covs=det_bbox_covs,
            bbox_covs=self.fixed_cov_coeff * torch.eye(4).to(img).unsqueeze(0).repeat(det_bboxes.shape[0], 1, 1),
            labels=det_labels,
            frame_id=frame_id,
            rescale=rescale,
            **kwargs)

        track_results = outs2results(
            bboxes=track_bboxes,
            bbox_covs=track_bbox_covs,
            labels=track_labels,
            ids=track_ids,
            num_classes=num_classes)
        det_results = outs2results(
            bboxes=det_bboxes, labels=det_labels, num_classes=num_classes)

        return dict(
            det_bboxes=det_results['bbox_results'],
            track_bboxes=track_results['bbox_results'],
            track_bbox_covs=track_results['bbox_cov_results'])