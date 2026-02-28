# -*- coding: utf-8 -*-
"""Core LightGlue utilities — no WBIA dependencies.

This module contains all logic that can be tested standalone:
configuration, caching, model management, feature conversion, and scoring.
"""
from __future__ import absolute_import, division, print_function

import collections
import json
import os
import threading

import cv2
import numpy as np
import torch

from lightglue import ALIKED, LightGlue
from lightglue.utils import numpy_image_to_torch, rbd

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    'model_name': 'aliked-n16',
    'max_num_keypoints': 2048,
    'detection_threshold': 0.0,
    'depth_confidence': 0.95,
    'width_confidence': 0.99,
    'filter_threshold': 0.1,
    'image_size': 512,
    'feature_cache_max_size': 10000,
}


def _load_config(config_path=None):
    """Load config from JSON file, falling back to defaults."""
    config = DEFAULT_CONFIG.copy()
    if config_path is None:
        config_path = '/v_config/lightglue/config.json'
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            config.update(user_config)
            print('LightGlue config loaded from %s' % config_path)
        except (json.JSONDecodeError, IOError) as e:
            print('Warning: could not load config from %s: %s' % (config_path, e))
    return config


# ---------------------------------------------------------------------------
# LRU Feature Cache
# ---------------------------------------------------------------------------

class _LRUFeatureCache:
    """Simple LRU cache for ALIKED features, keyed by (aid, config_path)."""

    def __init__(self, max_size=10000):
        self._cache = collections.OrderedDict()
        self._max_size = max_size

    def get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
        self._cache[key] = value

    def __contains__(self, key):
        return key in self._cache

    def __len__(self):
        return len(self._cache)

    def clear(self):
        self._cache.clear()


GLOBAL_FEATURE_CACHE = _LRUFeatureCache()


# ---------------------------------------------------------------------------
# Model singletons
# ---------------------------------------------------------------------------

_EXTRACTOR = None
_MATCHER = None
_DEVICE = None
_CONFIG = None
_CONFIG_PATH = None
_MODEL_LOCK = threading.Lock()


def _get_models(config_path=None):
    """Lazy-load ALIKED extractor and LightGlue matcher (singleton).

    Thread-safe. If called with a different config_path than was used to
    load the current models, the models are reloaded.
    """
    global _EXTRACTOR, _MATCHER, _DEVICE, _CONFIG, _CONFIG_PATH

    if _EXTRACTOR is not None and _MATCHER is not None:
        if config_path == _CONFIG_PATH:
            return _EXTRACTOR, _MATCHER, _DEVICE, _CONFIG
        print('LightGlue config changed (%s -> %s), reloading models'
              % (_CONFIG_PATH, config_path))

    with _MODEL_LOCK:
        # Double-check after acquiring lock
        if _EXTRACTOR is not None and config_path == _CONFIG_PATH:
            return _EXTRACTOR, _MATCHER, _DEVICE, _CONFIG

        config = _load_config(config_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        extractor = ALIKED(
            model_name=config['model_name'],
            max_num_keypoints=config['max_num_keypoints'],
            detection_threshold=config['detection_threshold'],
        ).eval().to(device)

        matcher = LightGlue(
            features='aliked',
            depth_confidence=config['depth_confidence'],
            width_confidence=config['width_confidence'],
            filter_threshold=config['filter_threshold'],
        ).eval().to(device)

        _EXTRACTOR = extractor
        _MATCHER = matcher
        _DEVICE = device
        _CONFIG = config
        _CONFIG_PATH = config_path

        # Update cache size from config
        GLOBAL_FEATURE_CACHE._max_size = config.get(
            'feature_cache_max_size', 10000
        )

        print('LightGlue models loaded on %s' % device)
        return _EXTRACTOR, _MATCHER, _DEVICE, _CONFIG


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _chip_to_tensor(chip):
    """Convert a WBIA annotation chip (BGR uint8 numpy) to a torch tensor.

    Converts BGR -> RGB and normalizes to [0, 1] float32 CHW tensor.
    Resizing is handled by extractor.extract() to avoid double-resize.
    """
    image = cv2.cvtColor(chip, cv2.COLOR_BGR2RGB)
    return numpy_image_to_torch(image)


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def _features_to_torch(feats_dict, device):
    """Convert cached numpy features back to the torch dict LightGlue expects."""
    result = {
        'keypoints': torch.from_numpy(feats_dict['keypoints'])[None].to(device),
        'descriptors': torch.from_numpy(feats_dict['descriptors'])[None].to(device),
        'keypoint_scores': torch.from_numpy(feats_dict['keypoint_scores'])[None].to(device),
    }
    if 'image_size' in feats_dict and feats_dict['image_size'] is not None:
        result['image_size'] = torch.from_numpy(
            feats_dict['image_size']
        )[None].to(device)
    return result


def _match_pair_score(matcher, feats0_torch, feats1_torch):
    """Run LightGlue on a feature pair and return sum of match confidences."""
    with torch.no_grad():
        result = matcher({'image0': feats0_torch, 'image1': feats1_torch})
    result = rbd(result)
    # After rbd, 'scores' is a 1D tensor of per-match confidence values
    scores = result['scores']
    if not isinstance(scores, torch.Tensor) or scores.numel() == 0:
        return 0.0
    return float(scores.sum().item())
