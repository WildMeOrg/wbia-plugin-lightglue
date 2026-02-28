# -*- coding: utf-8 -*-
"""Tests for the _LRUFeatureCache class."""
import numpy as np
import pytest

from wbia_lightglue.core import _LRUFeatureCache


class TestLRUFeatureCache:

    def test_put_and_get(self):
        cache = _LRUFeatureCache(max_size=10)
        cache.put('a', 1)
        assert cache.get('a') == 1

    def test_get_missing_returns_none(self):
        cache = _LRUFeatureCache(max_size=10)
        assert cache.get('missing') is None

    def test_contains(self):
        cache = _LRUFeatureCache(max_size=10)
        cache.put('a', 1)
        assert 'a' in cache
        assert 'b' not in cache

    def test_len(self):
        cache = _LRUFeatureCache(max_size=10)
        assert len(cache) == 0
        cache.put('a', 1)
        cache.put('b', 2)
        assert len(cache) == 2

    def test_eviction_at_max_size(self):
        cache = _LRUFeatureCache(max_size=3)
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)
        assert len(cache) == 3

        # Adding a 4th item should evict 'a' (least recently used)
        cache.put('d', 4)
        assert len(cache) == 3
        assert cache.get('a') is None
        assert cache.get('b') == 2
        assert cache.get('d') == 4

    def test_access_refreshes_lru_order(self):
        cache = _LRUFeatureCache(max_size=3)
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)

        # Access 'a' to make it recently used
        cache.get('a')

        # Adding 'd' should evict 'b' (now least recently used), not 'a'
        cache.put('d', 4)
        assert cache.get('a') == 1
        assert cache.get('b') is None

    def test_put_updates_existing_key(self):
        cache = _LRUFeatureCache(max_size=3)
        cache.put('a', 1)
        cache.put('a', 2)
        assert cache.get('a') == 2
        assert len(cache) == 1

    def test_clear(self):
        cache = _LRUFeatureCache(max_size=10)
        cache.put('a', 1)
        cache.put('b', 2)
        cache.clear()
        assert len(cache) == 0
        assert cache.get('a') is None

    def test_tuple_keys(self):
        """Cache keys are (aid, config_path) tuples in production."""
        cache = _LRUFeatureCache(max_size=10)
        cache.put((1, None), 'features_default')
        cache.put((1, '/custom/config.json'), 'features_custom')
        assert cache.get((1, None)) == 'features_default'
        assert cache.get((1, '/custom/config.json')) == 'features_custom'
        assert len(cache) == 2

    def test_stores_numpy_arrays(self):
        """Verify cache works with dict-of-numpy values (production pattern)."""
        cache = _LRUFeatureCache(max_size=5)
        feats = {
            'keypoints': np.random.rand(100, 2).astype(np.float32),
            'descriptors': np.random.rand(100, 128).astype(np.float32),
            'keypoint_scores': np.random.rand(100).astype(np.float32),
        }
        cache.put((42, None), feats)
        retrieved = cache.get((42, None))
        np.testing.assert_array_equal(retrieved['keypoints'], feats['keypoints'])
        np.testing.assert_array_equal(retrieved['descriptors'], feats['descriptors'])

    def test_max_size_one(self):
        cache = _LRUFeatureCache(max_size=1)
        cache.put('a', 1)
        cache.put('b', 2)
        assert len(cache) == 1
        assert cache.get('a') is None
        assert cache.get('b') == 2
