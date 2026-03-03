# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import tqdm
import utool as ut
import vtool as vt
import wbia
from wbia import dtool as dt
from wbia.constants import ANNOTATION_TABLE
from wbia.control import controller_inject

from lightglue.utils import rbd
from wbia_lightglue.core import (
    DEFAULT_CONFIG,  # noqa: F401
    GLOBAL_FEATURE_CACHE,
    _LRUFeatureCache,  # noqa: F401
    _chip_to_tensor,
    _features_to_torch,
    _get_models,
    _load_config,  # noqa: F401
    _match_pair_score,
)

(print, rrr, profile) = ut.inject2(__name__)

_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)

register_api = controller_inject.get_wbia_flask_api(__name__)
register_route = controller_inject.get_wbia_flask_route(__name__)

register_preproc_annot = controller_inject.register_preprocs['annot']


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


@register_ibs_method
def lightglue_compute_features(ibs, aid_list, config_path=None):
    """Extract ALIKED keypoints and descriptors for a list of annotations.

    Returns:
        list of dicts with keys 'keypoints', 'descriptors', 'keypoint_scores',
        each as numpy arrays.
    """
    extractor, _, device, config = _get_models(config_path)
    image_size = config['image_size']

    chips = ibs.get_annot_chips(aid_list)

    results = []
    with torch.inference_mode():
        for chip in chips:
            img_tensor = _chip_to_tensor(chip).to(device)
            feats = extractor.extract(img_tensor, resize=image_size)
            feats = rbd(feats)  # remove batch dim
            results.append({
                'keypoints': feats['keypoints'].cpu().numpy(),
                'descriptors': feats['descriptors'].cpu().numpy(),
                'keypoint_scores': feats['keypoint_scores'].cpu().numpy(),
                'image_size': feats['image_size'].cpu().numpy(),
            })

    return results


@register_ibs_method
def lightglue_features(ibs, aid_list, config_path=None, use_depc=True):
    """Get ALIKED features for annotations, using LRU cache when available.

    Cache keys include config_path so different configs don't collide.
    """
    cache = GLOBAL_FEATURE_CACHE
    dirty_aids = [aid for aid in aid_list if (aid, config_path) not in cache]

    if len(dirty_aids) > 0:
        print('Computing %d non-cached LightGlue features' % len(dirty_aids))
        if use_depc:
            config_map = {'config_path': config_path}
            colnames = ['keypoints', 'descriptors', 'keypoint_scores', 'image_size']
            results = ibs.depc_annot.get(
                'LightGlueFeatures', dirty_aids, colnames, config_map
            )
            for aid, (kpts, desc, scores, img_size) in zip(dirty_aids, results):
                cache.put((aid, config_path), {
                    'keypoints': kpts,
                    'descriptors': desc,
                    'keypoint_scores': scores,
                    'image_size': img_size,
                })
        else:
            dirty_features = lightglue_compute_features(
                ibs, dirty_aids, config_path
            )
            for aid, feats in zip(dirty_aids, dirty_features):
                cache.put((aid, config_path), feats)

    return [cache.get((aid, config_path)) for aid in aid_list]


# ---------------------------------------------------------------------------
# Depc table: LightGlueFeatures (per-annotation)
# ---------------------------------------------------------------------------

class LightGlueFeatureConfig(dt.Config):  # NOQA
    _param_info_list = [
        ut.ParamInfo('config_path', default=None),
    ]


@register_preproc_annot(
    tablename='LightGlueFeatures',
    parents=[ANNOTATION_TABLE],
    colnames=['keypoints', 'descriptors', 'keypoint_scores', 'image_size'],
    coltypes=[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    configclass=LightGlueFeatureConfig,
    fname='lightglue',
    chunksize=64,
)
@register_ibs_method
def lightglue_features_depc(depc, aid_list, config=None):
    ibs = depc.controller
    features_list = lightglue_compute_features(
        ibs, aid_list, config_path=config['config_path']
    )
    for feats in features_list:
        yield (
            feats['keypoints'],
            feats['descriptors'],
            feats['keypoint_scores'],
            feats['image_size'],
        )


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

@register_ibs_method
def lightglue_match_scores(ibs, qaid, daid_list, config_path=None):
    """Compute LightGlue match scores between a query and database annotations.

    Returns:
        list of float scores, one per daid in daid_list.
    """
    _, matcher, device, _ = _get_models(config_path)

    # Get cached features
    all_aids = [qaid] + list(daid_list)
    all_feats = ibs.lightglue_features(all_aids, config_path=config_path)
    query_feats = all_feats[0]
    db_feats_list = all_feats[1:]

    # Batch-transfer all features to GPU at once to amortize PCIe overhead
    with torch.inference_mode():
        query_torch = _features_to_torch(query_feats, device)
        db_torch_list = [_features_to_torch(f, device) for f in db_feats_list]

    scores = []
    for db_torch in db_torch_list:
        score = _match_pair_score(matcher, query_torch, db_torch)
        scores.append(score)

    return scores


# ---------------------------------------------------------------------------
# Depc table: LightGlue (pairwise similarity)
# ---------------------------------------------------------------------------

class LightGlueConfig(dt.Config):  # NOQA
    def get_param_info_list(self):
        return [
            ut.ParamInfo('config_path', None),
        ]


def get_match_results(depc, qaid_list, daid_list, score_list, config):
    """Convert table results into AnnotMatch objects.

    Uses np.sum for name-level aggregation (HotSpotter nsum-style) so that
    multiple annotations of the same individual accumulate evidence.
    """
    unique_qaids, groupxs = ut.group_indices(qaid_list)
    grouped_daids = ut.apply_grouping(daid_list, groupxs)
    grouped_scores = ut.apply_grouping(score_list, groupxs)

    ibs = depc.controller
    unique_qnids = ibs.get_annot_nids(unique_qaids)

    _iter = zip(unique_qaids, unique_qnids, grouped_daids, grouped_scores)
    for qaid, qnid, daids, scores in _iter:
        dnids = ibs.get_annot_nids(daids)

        annot_scores = np.array(scores)
        daid_list_ = np.array(daids)
        dnid_list_ = np.array(dnids)

        # Remove self-match and zero-score entries
        is_valid = (daid_list_ != qaid) & (annot_scores > 0)
        daid_list_ = daid_list_.compress(is_valid)
        dnid_list_ = dnid_list_.compress(is_valid)
        annot_scores = annot_scores.compress(is_valid)

        match_result = wbia.AnnotMatch()
        match_result.qaid = qaid
        match_result.qnid = qnid
        match_result.daid_list = daid_list_
        match_result.dnid_list = dnid_list_
        match_result._update_daid_index()
        match_result._update_unique_nid_index()

        grouped_annot_scores = vt.apply_grouping(
            annot_scores, match_result.name_groupxs
        )
        # nsum: sum of annotation scores per name (HotSpotter-style)
        name_scores = np.array([np.sum(s) for s in grouped_annot_scores])
        match_result.set_cannonical_name_score(annot_scores, name_scores)
        yield match_result


class LightGlueRequest(dt.base.VsOneSimilarityRequest):
    _symmetric = True
    _tablename = 'LightGlue'

    @ut.accepts_scalar_input
    def get_fmatch_overlayed_chip(request, aid_list, overlay=True, config=None):
        depc = request.depc
        ibs = depc.controller
        chips = ibs.get_annot_chips(aid_list)
        return chips

    def render_single_result(request, cm, aid, **kwargs):
        depc = request.depc
        ibs = depc.controller
        chips = request.get_fmatch_overlayed_chip([cm.qaid, aid])
        return vt.stack_image_list(chips)

    def postprocess_execute(request, table, parent_rowids, rowids, result_list):
        qaid_list, daid_list = list(zip(*parent_rowids))
        score_list = ut.take_column(result_list, 0)
        depc = request.depc
        config = request.config
        cm_list = list(
            get_match_results(depc, qaid_list, daid_list, score_list, config)
        )
        table.delete_rows(rowids)
        return cm_list

    def execute(request, *args, **kwargs):
        result_list = super(LightGlueRequest, request).execute(*args, **kwargs)
        qaids = kwargs.pop('qaids', None)
        if qaids is not None:
            result_list = [
                result for result in result_list if result.qaid in qaids
            ]
        return result_list


@register_preproc_annot(
    tablename='LightGlue',
    parents=[ANNOTATION_TABLE, ANNOTATION_TABLE],
    colnames=['score'],
    coltypes=[float],
    configclass=LightGlueConfig,
    requestclass=LightGlueRequest,
    fname='lightglue',
    rm_extern_on_delete=True,
    chunksize=None,
)
def wbia_plugin_lightglue(depc, qaid_list, daid_list, config):
    ibs = depc.controller

    qaids = list(set(qaid_list))
    daids = list(set(daid_list))

    qaid_score_dict = {}
    for qaid in tqdm.tqdm(qaids, desc='LightGlue matching'):
        scores = ibs.lightglue_match_scores(
            qaid,
            daids,
            config_path=config['config_path'],
        )
        qaid_score_dict[qaid] = dict(zip(daids, scores))

    for qaid, daid in zip(qaid_list, daid_list):
        if qaid == daid:
            daid_score = 0.0
        else:
            aid_score_dict = qaid_score_dict.get(qaid, {})
            daid_score = aid_score_dict.get(daid, 0.0)
        yield (daid_score,)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@register_ibs_method
def lightglue_evaluate(ibs, aid_list, config_path=None, ranks=(1, 5, 10, 20)):
    """Evaluate LightGlue matching on a set of annotations using 1-vs-all."""
    n = len(aid_list)
    print('Computing %d x %d pairwise LightGlue scores...' % (n, n))

    score_matrix = np.zeros((n, n), dtype=np.float64)
    for i, qaid in enumerate(tqdm.tqdm(aid_list, desc='Evaluating')):
        daids = [aid for aid in aid_list if aid != qaid]
        scores = ibs.lightglue_match_scores(qaid, daids, config_path=config_path)
        j = 0
        for k, daid in enumerate(aid_list):
            if daid == qaid:
                score_matrix[i, k] = 0.0
            else:
                score_matrix[i, k] = scores[j]
                j += 1

    db_labels = np.array(ibs.get_annot_name_rowids(aid_list))

    # Compute CMC ranks
    n_correct = {r: 0 for r in ranks}
    for i in range(n):
        query_label = db_labels[i]
        # Sort by score descending, exclude self by index (not label,
        # since other annotations may share the same individual label)
        order = np.argsort(-score_matrix[i])
        order_no_self = [idx for idx in order if idx != i]
        ordered_labels = db_labels[order_no_self]
        for r in ranks:
            if query_label in ordered_labels[:r]:
                n_correct[r] += 1

    print('** LightGlue Results **')
    for r in ranks:
        print('Rank-%-3d: %.1f%%' % (r, 100.0 * n_correct[r] / n))

    return n_correct[ranks[0]] / n


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia_lightglue._plugin --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()
    ut.doctest_funcs()
