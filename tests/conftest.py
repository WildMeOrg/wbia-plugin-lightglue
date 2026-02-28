# -*- coding: utf-8 -*-
"""Shared fixtures for wbia-plugin-lightglue tests."""
import os

import cv2
import numpy as np
import pytest
import torch

from lightglue import ALIKED, LightGlue
from lightglue.utils import numpy_image_to_torch, rbd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'LightGlue', 'assets')
IMAGE_PATH_1 = os.path.join(ASSETS_DIR, 'sacre_coeur1.jpg')
IMAGE_PATH_2 = os.path.join(ASSETS_DIR, 'sacre_coeur2.jpg')


def _resolve(path):
    """Resolve and verify a path exists."""
    p = os.path.realpath(path)
    if not os.path.exists(p):
        pytest.skip(f'Test asset not found: {p}')
    return p


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(scope='session')
def extractor(device):
    return ALIKED(
        model_name='aliked-n16',
        max_num_keypoints=512,
        detection_threshold=0.0,
    ).eval().to(device)


@pytest.fixture(scope='session')
def matcher(device):
    return LightGlue(
        features='aliked',
        depth_confidence=0.95,
        width_confidence=0.99,
    ).eval().to(device)


@pytest.fixture(scope='session')
def image_pair():
    """Load the sacre coeur image pair as RGB numpy arrays."""
    p1 = _resolve(IMAGE_PATH_1)
    p2 = _resolve(IMAGE_PATH_2)
    img1 = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB)
    return img1, img2


@pytest.fixture(scope='session')
def feature_pair(extractor, image_pair, device):
    """Extract ALIKED features for both images (session-cached)."""
    img1, img2 = image_pair
    with torch.no_grad():
        t1 = numpy_image_to_torch(img1).to(device)
        t2 = numpy_image_to_torch(img2).to(device)
        feats1 = rbd(extractor.extract(t1, resize=512))
        feats2 = rbd(extractor.extract(t2, resize=512))
    return feats1, feats2


@pytest.fixture
def synthetic_chip():
    """A small synthetic BGR image (like a WBIA chip)."""
    return np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
