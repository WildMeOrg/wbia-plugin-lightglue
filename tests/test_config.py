# -*- coding: utf-8 -*-
"""Tests for configuration loading."""
import json

import pytest

from wbia_lightglue.core import DEFAULT_CONFIG, _load_config


class TestLoadConfig:

    def test_defaults_when_no_file(self, tmp_path):
        config = _load_config(str(tmp_path / 'nonexistent.json'))
        for key in DEFAULT_CONFIG:
            assert key in config
            assert config[key] == DEFAULT_CONFIG[key]

    def test_loads_from_json(self, tmp_path):
        config_file = tmp_path / 'config.json'
        config_file.write_text(json.dumps({
            'model_name': 'aliked-n32',
            'max_num_keypoints': 1024,
        }))
        config = _load_config(str(config_file))
        assert config['model_name'] == 'aliked-n32'
        assert config['max_num_keypoints'] == 1024
        # Unset keys should still have defaults
        assert config['depth_confidence'] == DEFAULT_CONFIG['depth_confidence']

    def test_invalid_json_falls_back(self, tmp_path):
        config_file = tmp_path / 'bad.json'
        config_file.write_text('not valid json {{{')
        config = _load_config(str(config_file))
        # Should fall back to defaults
        assert config['model_name'] == DEFAULT_CONFIG['model_name']

    def test_default_path_when_none(self):
        # When config_path is None, it tries /v_config/lightglue/config.json
        # which doesn't exist locally, so defaults are used
        config = _load_config(None)
        assert config['model_name'] == DEFAULT_CONFIG['model_name']

    def test_feature_cache_max_size_in_defaults(self):
        assert 'feature_cache_max_size' in DEFAULT_CONFIG
        assert DEFAULT_CONFIG['feature_cache_max_size'] == 10000
