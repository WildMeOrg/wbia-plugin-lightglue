# -*- coding: utf-8 -*-
"""Tests for model singleton loading and thread safety."""
import threading

import pytest
import torch

import wbia_lightglue.core as core


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset model singletons before each test."""
    core._EXTRACTOR = None
    core._MATCHER = None
    core._DEVICE = None
    core._CONFIG = None
    core._CONFIG_PATH = None
    core.GLOBAL_FEATURE_CACHE.clear()
    yield
    core._EXTRACTOR = None
    core._MATCHER = None
    core._DEVICE = None
    core._CONFIG = None
    core._CONFIG_PATH = None
    core.GLOBAL_FEATURE_CACHE.clear()


class TestGetModels:

    def test_loads_models(self):
        extractor, matcher, device, config = core._get_models()
        assert extractor is not None
        assert matcher is not None
        assert isinstance(device, torch.device)
        assert isinstance(config, dict)

    def test_singleton_returns_same_objects(self):
        ext1, mat1, _, _ = core._get_models()
        ext2, mat2, _, _ = core._get_models()
        assert ext1 is ext2
        assert mat1 is mat2

    def test_config_path_tracked(self):
        core._get_models(config_path=None)
        assert core._CONFIG_PATH is None

    def test_different_config_reloads(self, tmp_path):
        import json
        ext1, _, _, _ = core._get_models(config_path=None)

        config_file = tmp_path / 'config.json'
        config_file.write_text(json.dumps({'image_size': 256}))

        ext2, _, _, config2 = core._get_models(config_path=str(config_file))
        assert config2['image_size'] == 256
        assert ext1 is not ext2

    def test_thread_safety(self):
        """Multiple threads loading models should not crash."""
        results = []
        errors = []

        def load_models():
            try:
                ext, mat, dev, cfg = core._get_models()
                results.append((id(ext), id(mat)))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=load_models) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f'Thread errors: {errors}'
        assert len(results) == 8
        # All threads should get the same singleton
        assert len(set(results)) == 1

    def test_cache_size_updated_from_config(self, tmp_path):
        import json
        config_file = tmp_path / 'config.json'
        config_file.write_text(json.dumps({'feature_cache_max_size': 500}))
        core._get_models(config_path=str(config_file))
        assert core.GLOBAL_FEATURE_CACHE._max_size == 500
