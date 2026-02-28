# -*- coding: utf-8 -*-
"""Tests for LightGlue matching and scoring."""
import numpy as np
import pytest
import torch

from lightglue.utils import rbd
from wbia_lightglue.core import _features_to_torch, _match_pair_score


class TestMatchPairScore:

    def test_matching_same_scene_positive_score(self, matcher, feature_pair, device):
        """Two views of the same scene should have a positive match score."""
        feats1, feats2 = feature_pair
        feats1_np = {k: v.cpu().numpy() for k, v in feats1.items()}
        feats2_np = {k: v.cpu().numpy() for k, v in feats2.items()}

        f1 = _features_to_torch(feats1_np, device)
        f2 = _features_to_torch(feats2_np, device)
        score = _match_pair_score(matcher, f1, f2)

        assert isinstance(score, float)
        assert score > 0.0

    def test_matching_produces_many_matches(self, matcher, feature_pair, device):
        """Sacre Coeur pair should produce many confident matches."""
        feats1, feats2 = feature_pair
        feats1_np = {k: v.cpu().numpy() for k, v in feats1.items()}
        feats2_np = {k: v.cpu().numpy() for k, v in feats2.items()}

        f1 = _features_to_torch(feats1_np, device)
        f2 = _features_to_torch(feats2_np, device)

        with torch.no_grad():
            result = rbd(matcher({'image0': f1, 'image1': f2}))

        n_matches = result['matches'].shape[0]
        assert n_matches > 50, f'Expected many matches for same scene, got {n_matches}'

    def test_self_match_highest_score(self, matcher, feature_pair, device):
        """Matching an image to itself should produce the highest score."""
        feats1, feats2 = feature_pair
        feats1_np = {k: v.cpu().numpy() for k, v in feats1.items()}
        feats2_np = {k: v.cpu().numpy() for k, v in feats2.items()}

        f1 = _features_to_torch(feats1_np, device)
        f2 = _features_to_torch(feats2_np, device)

        self_score = _match_pair_score(matcher, f1, f1)
        cross_score = _match_pair_score(matcher, f1, f2)

        assert self_score > cross_score

    def test_score_is_symmetric(self, matcher, feature_pair, device):
        """score(A, B) should equal score(B, A)."""
        feats1, feats2 = feature_pair
        feats1_np = {k: v.cpu().numpy() for k, v in feats1.items()}
        feats2_np = {k: v.cpu().numpy() for k, v in feats2.items()}

        f1 = _features_to_torch(feats1_np, device)
        f2 = _features_to_torch(feats2_np, device)

        score_ab = _match_pair_score(matcher, f1, f2)
        score_ba = _match_pair_score(matcher, f2, f1)

        assert abs(score_ab - score_ba) < 0.01, (
            f'Scores should be symmetric: {score_ab} vs {score_ba}'
        )

    def test_empty_features_return_zero(self, matcher, device):
        """If one image has zero keypoints, score should be 0."""
        empty = {
            'keypoints': torch.zeros((1, 0, 2), device=device),
            'descriptors': torch.zeros((1, 0, 128), device=device),
            'keypoint_scores': torch.zeros((1, 0), device=device),
            'image_size': torch.tensor([[512, 512]], device=device),
        }
        nonempty = {
            'keypoints': torch.rand((1, 10, 2), device=device) * 512,
            'descriptors': torch.randn((1, 10, 128), device=device),
            'keypoint_scores': torch.ones((1, 10), device=device),
            'image_size': torch.tensor([[512, 512]], device=device),
        }

        score = _match_pair_score(matcher, empty, nonempty)
        assert score == 0.0


class TestFeaturesToTorch:

    def test_adds_batch_dimension(self, device):
        feats_np = {
            'keypoints': np.random.rand(50, 2).astype(np.float32),
            'descriptors': np.random.rand(50, 128).astype(np.float32),
            'keypoint_scores': np.random.rand(50).astype(np.float32),
            'image_size': np.array([512, 512], dtype=np.float32),
        }
        result = _features_to_torch(feats_np, device)

        assert result['keypoints'].shape == (1, 50, 2)
        assert result['descriptors'].shape == (1, 50, 128)
        assert result['keypoint_scores'].shape == (1, 50)

    def test_tensors_on_correct_device(self, device):
        feats_np = {
            'keypoints': np.random.rand(10, 2).astype(np.float32),
            'descriptors': np.random.rand(10, 128).astype(np.float32),
            'keypoint_scores': np.random.rand(10).astype(np.float32),
            'image_size': np.array([512, 512], dtype=np.float32),
        }
        result = _features_to_torch(feats_np, device)
        assert result['keypoints'].device.type == device.type
        assert result['descriptors'].device.type == device.type

    def test_handles_missing_image_size(self, device):
        feats_np = {
            'keypoints': np.random.rand(10, 2).astype(np.float32),
            'descriptors': np.random.rand(10, 128).astype(np.float32),
            'keypoint_scores': np.random.rand(10).astype(np.float32),
        }
        result = _features_to_torch(feats_np, device)
        assert 'image_size' not in result
