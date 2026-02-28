# -*- coding: utf-8 -*-
"""Tests for ALIKED feature extraction pipeline."""
import numpy as np
import pytest
import torch

from wbia_lightglue.core import _chip_to_tensor


class TestChipToTensor:

    def test_bgr_to_rgb_conversion(self):
        """_chip_to_tensor should convert BGR -> RGB."""
        chip = np.zeros((100, 100, 3), dtype=np.uint8)
        chip[:, :, 0] = 255  # B channel = 255
        chip[:, :, 1] = 0    # G channel = 0
        chip[:, :, 2] = 0    # R channel = 0

        tensor = _chip_to_tensor(chip)

        # numpy_image_to_torch returns [1, C, H, W]
        # After BGR->RGB, channel 0 (R) should be 0, channel 2 (B) should be ~1.0
        assert tensor.shape[-3] == 3  # C dimension
        assert tensor[..., 0, 0, 0].item() < 0.01   # R channel ~ 0
        assert tensor[..., 2, 0, 0].item() > 0.99    # B channel ~ 1

    def test_output_shape(self, synthetic_chip):
        tensor = _chip_to_tensor(synthetic_chip)
        # numpy_image_to_torch returns [C, H, W] (3D)
        assert tensor.ndim >= 3
        assert tensor.shape[-3] == 3  # C dimension

    def test_output_range(self, synthetic_chip):
        tensor = _chip_to_tensor(synthetic_chip)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0
        assert tensor.dtype == torch.float32


class TestFeatureExtraction:

    def test_extract_returns_expected_keys(self, feature_pair):
        feats1, _ = feature_pair
        assert 'keypoints' in feats1
        assert 'descriptors' in feats1
        assert 'keypoint_scores' in feats1
        assert 'image_size' in feats1

    def test_keypoints_shape(self, feature_pair):
        feats1, _ = feature_pair
        kpts = feats1['keypoints']
        assert kpts.ndim == 2
        assert kpts.shape[1] == 2  # (N, 2) — x, y coordinates

    def test_descriptors_shape(self, feature_pair):
        feats1, _ = feature_pair
        desc = feats1['descriptors']
        n_keypoints = feats1['keypoints'].shape[0]
        assert desc.ndim == 2
        assert desc.shape[0] == n_keypoints
        assert desc.shape[1] == 128  # ALIKED-N16 descriptor dim

    def test_scores_shape(self, feature_pair):
        feats1, _ = feature_pair
        scores = feats1['keypoint_scores']
        n_keypoints = feats1['keypoints'].shape[0]
        assert scores.ndim == 1
        assert scores.shape[0] == n_keypoints

    def test_max_keypoints_respected(self, feature_pair):
        feats1, feats2 = feature_pair
        # extractor configured with max_num_keypoints=512
        assert feats1['keypoints'].shape[0] <= 512
        assert feats2['keypoints'].shape[0] <= 512

    def test_both_images_produce_features(self, feature_pair):
        feats1, feats2 = feature_pair
        assert feats1['keypoints'].shape[0] > 0
        assert feats2['keypoints'].shape[0] > 0
