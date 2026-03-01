# wbia-plugin-lightglue

WBIA plugin for local feature matching using [ALIKED](https://arxiv.org/abs/2304.03608) keypoint detection + [LightGlue](https://arxiv.org/abs/2306.13643) matching. Provides pairwise annotation similarity scores to the Wildbook IA platform.

## How it works

1. **Feature extraction** -- ALIKED detects keypoints and computes 128-dim descriptors for each annotation chip.
2. **Matching** -- LightGlue matches keypoints between an image pair using self+cross attention with adaptive depth/width pruning.
3. **Scoring** -- The sum of per-match confidence values becomes the raw similarity score for that pair.

Features are cached in an LRU cache (default 10,000 entries) so repeated queries against the same annotation avoid re-extraction. Models are lazy-loaded as singletons and optionally compiled with `torch.compile` on CUDA for faster inference.

## Directory structure

```
wbia-plugin-lightglue/
├── wbia_lightglue/
│   ├── __init__.py        # Package init; imports _plugin to register with WBIA
│   ├── _plugin.py         # WBIA integration: ibs methods, depc tables, Flask routes
│   └── core.py            # Standalone utilities: config, caching, model loading, scoring
├── tests/
│   ├── conftest.py        # Shared pytest fixtures
│   ├── test_config.py     # Config loading and defaults
│   ├── test_feature_extraction.py
│   ├── test_lru_cache.py  # LRU cache behavior
│   ├── test_matching.py   # Match scoring logic
│   ├── test_model_loading.py
│   └── test_scoring.py    # Score aggregation
├── setup.py               # Package metadata and dependencies
├── requirements.txt       # Pin file for dependencies
├── pytest.ini             # Test runner configuration
└── .gitignore
```

## Files created on disk

| Location | What | When |
|----------|------|------|
| `~/.cache/torch/hub/checkpoints/` | ALIKED and LightGlue model weights (`.pth` files, ~50 MB total) | First model load; downloaded from GitHub releases via `torch.hub` |
| `/v_config/lightglue/config.json` | Optional JSON config override | User-created; never written by the plugin |
| WBIA database `_ibsdb/_wbia_cache/` | Depc tables: `LightGlueFeatures` (per-annotation) and `LightGlue` (pairwise scores) | On first query through the depc pipeline |

**In-memory only (not persisted to disk):**

| Item | Description |
|------|-------------|
| LRU feature cache | Up to `feature_cache_max_size` (default 10,000) ALIKED feature dicts keyed by `(aid, config_path)` |
| Model singletons | One ALIKED extractor + one LightGlue matcher, loaded once per config |

## Configuration

The plugin reads configuration from a JSON file, falling back to built-in defaults for any missing key.

**Config file location:** `/v_config/lightglue/config.json` (override via `config_path` parameter)

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | string | `"aliked-n16"` | ALIKED model variant |
| `max_num_keypoints` | int | `2048` | Max keypoints per image (higher = more accurate, slower) |
| `detection_threshold` | float | `0.0` | Keypoint detection confidence threshold |
| `depth_confidence` | float | `0.95` | Early stopping threshold (-1 to disable). Lower = faster, less accurate |
| `width_confidence` | float | `0.99` | Point pruning threshold (-1 to disable). Lower = faster, less accurate |
| `filter_threshold` | float | `0.1` | Minimum match confidence to accept. Higher = fewer but stronger matches |
| `image_size` | int | `512` | Resize long edge before extraction |
| `feature_cache_max_size` | int | `10000` | Max annotations held in the in-memory LRU cache |

### Example config file

```json
{
    "max_num_keypoints": 4096,
    "image_size": 1024,
    "depth_confidence": -1,
    "width_confidence": -1
}
```

## WBIA integration

### Registered methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `ibs.lightglue_compute_features` | `(aid_list, config_path=None)` | Extract ALIKED features for a list of annotations |
| `ibs.lightglue_features` | `(aid_list, config_path=None, use_depc=True)` | Get features with LRU caching and optional depc backing |
| `ibs.lightglue_match_scores` | `(qaid, daid_list, config_path=None)` | Compute raw match scores (query vs each database annotation) |
| `ibs.lightglue_evaluate` | `(aid_list, config_path=None, ranks=(1,5,10,20))` | Run 1-vs-all CMC evaluation on a set of annotations |

### Depc tables

| Table | Parents | Columns | Description |
|-------|---------|---------|-------------|
| `LightGlueFeatures` | `ANNOTATION` | `keypoints`, `descriptors`, `keypoint_scores`, `image_size` | Per-annotation ALIKED features (cached to SQLite) |
| `LightGlue` | `ANNOTATION`, `ANNOTATION` | `score` | Pairwise similarity score |

### Request parameters (via Wildbook HTTP)

When Wildbook sends an identification request to WBIA, it specifies a pipeline (e.g., `"LightGlue"`) and can include configuration overrides. These flow through the `LightGlueConfig` depc config class:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `config_path` | `None` | Path to a JSON config file on the WBIA server |

### Web call examples

**Minimal -- use all defaults:**

```bash
# Trigger LightGlue identification via WBIA's query endpoint
curl -X POST http://WBIA_HOST:5000/api/review/query/chip/best/ \
  -F 'query_annot_uuid_list=["UUID_HERE"]' \
  -F 'database_annot_uuid_list=["UUID1","UUID2","UUID3"]' \
  -F 'pipeline_root=LightGlue'
```

**With custom config file:**

```bash
curl -X POST http://WBIA_HOST:5000/api/review/query/chip/best/ \
  -F 'query_annot_uuid_list=["UUID_HERE"]' \
  -F 'database_annot_uuid_list=["UUID1","UUID2","UUID3"]' \
  -F 'pipeline_root=LightGlue' \
  -F 'config_path=/v_config/lightglue/highres.json'
```

**Internal Python (within WBIA):**

```python
# Direct method call
scores = ibs.lightglue_match_scores(qaid, daid_list)

# Via depc pipeline (results are cached)
cm_list = ibs.depc_annot.request('LightGlue', qaid_list, daid_list)

# Evaluate accuracy on a dataset
rank1 = ibs.lightglue_evaluate(aid_list)
```

## Installation

### Prerequisites

- Python >= 3.8
- CUDA-capable GPU (strongly recommended; CPU works but is slow)
- A running WBIA instance
- The `lightglue` package (upstream LightGlue library)

### Install

```bash
# 1. Install the upstream LightGlue library
cd /path/to/LightGlue
pip install -e .

# 2. Install this plugin
cd /path/to/wbia-plugin-lightglue
pip install -e .
```

WBIA discovers the plugin automatically on import. Verify by checking for registered tables:

```python
import wbia_lightglue
print(wbia_lightglue.__version__)  # '0.1.0'
```

### Docker / WBIA container

When building a WBIA Docker image, add both packages to the Dockerfile:

```dockerfile
# Install LightGlue (upstream library)
RUN git clone https://github.com/cvg/LightGlue.git /opt/LightGlue \
    && pip install -e /opt/LightGlue

# Install WBIA LightGlue plugin
COPY wbia-plugin-lightglue /opt/wbia-plugin-lightglue
RUN pip install -e /opt/wbia-plugin-lightglue
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >= 2.0.0 | Neural network inference, torch.compile |
| `torchvision` | >= 0.15.0 | Image transforms |
| `kornia` | >= 0.6.11 | ALIKED/DISK feature extraction |
| `numpy` | any | Array operations |
| `opencv-python-headless` | any | Image I/O, BGR/RGB conversion |
| `tqdm` | any | Progress bars |
| `lightglue` | (local) | Upstream ALIKED + LightGlue models |

### Running tests

```bash
cd /path/to/wbia-plugin-lightglue
pip install pytest
pytest
```

Tests use mocks and do not require a running WBIA instance or GPU.

## Performance notes

- **GPU (RTX 3080):** ~17 ms per pair at 2048 keypoints with `torch.compile` and mixed precision
- **CPU:** ~50 ms per pair at 512 keypoints
- First query is slower due to model loading and JIT compilation
- The LRU feature cache eliminates redundant extraction when the same annotation appears in multiple queries
- Model weights are downloaded once and cached by PyTorch in `~/.cache/torch/hub/checkpoints/`

## Score interpretation

Raw LightGlue scores are **unbounded positive floats** (sum of per-match confidences). Higher is better. Typical ranges:

- **0--5**: Weak or no match (different individuals, different viewpoints)
- **10--30**: Moderate match
- **50+**: Strong match (same individual, similar pose)

These raw scores are not normalized to [0, 1]. The [Hybrid plugin](../wbia-plugin-hybrid/) applies sigmoid normalization when fusing with MiewID scores.
