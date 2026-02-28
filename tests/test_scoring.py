# -*- coding: utf-8 -*-
"""Tests for score aggregation and evaluation logic."""
import numpy as np
import pytest


class TestNsumAggregation:
    """Test that name-level scoring uses np.sum (HotSpotter nsum)."""

    def test_sum_aggregation_accumulates(self):
        """Multiple annotations of the same individual should sum."""
        # Simulate: query matched against 4 database annotations
        # Two from individual A (scores 5.0, 3.0) and two from B (scores 7.0, 1.0)
        # nsum: A = 5+3 = 8, B = 7+1 = 8 (tie)
        # With np.max: A = 5, B = 7 (B wins) -- NOT what we want
        annot_scores = np.array([5.0, 3.0, 7.0, 1.0])
        dnid_list = np.array([100, 100, 200, 200])

        # Group by individual (replicating what vtool.apply_grouping does)
        unique_nids = np.unique(dnid_list)
        name_scores = []
        for nid in unique_nids:
            mask = dnid_list == nid
            name_scores.append(np.sum(annot_scores[mask]))
        name_scores = np.array(name_scores)

        # Both individuals should have score 8
        assert len(name_scores) == 2
        for s in name_scores:
            assert s == 8.0

    def test_sum_beats_max_for_multi_sighting(self):
        """nsum should favor individuals with many moderate matches over one strong match."""
        # Individual A: 3 annotations matched at 4.0 each → nsum = 12
        # Individual B: 1 annotation matched at 10.0 → nsum = 10
        # A should win with nsum, B would win with max
        scores_a = [4.0, 4.0, 4.0]
        scores_b = [10.0]

        assert np.sum(scores_a) > np.sum(scores_b)
        assert np.max(scores_a) < np.max(scores_b)


class TestCMCEvaluation:
    """Test the CMC/mAP evaluation protocol used in benchmarking."""

    def _make_score_matrix_and_labels(self):
        """Create a small score matrix with known correct ranking.

        4 annotations: [A0, A1, B0, B1]
        Individual A has annotations 0, 1
        Individual B has annotations 2, 3
        Scores designed so same-individual pairs score highest.
        """
        labels = np.array(['A', 'A', 'B', 'B'])
        scores = np.array([
            [0.0, 9.0, 2.0, 1.0],  # A0: best match is A1 (9.0) ✓
            [9.0, 0.0, 1.0, 2.0],  # A1: best match is A0 (9.0) ✓
            [1.0, 2.0, 0.0, 8.0],  # B0: best match is B1 (8.0) ✓
            [2.0, 1.0, 8.0, 0.0],  # B1: best match is B0 (8.0) ✓
        ], dtype=np.float64)
        return scores, labels

    def test_perfect_rank1(self):
        """When all queries have correct match at rank 1, CMC@1 = 100%."""
        scores, labels = self._make_score_matrix_and_labels()
        n = len(labels)

        n_correct = 0
        for i in range(n):
            order = np.argsort(-scores[i])
            order_no_self = [idx for idx in order if idx != i]
            if labels[order_no_self[0]] == labels[i]:
                n_correct += 1

        assert n_correct == n  # 100% Rank-1

    def test_imperfect_ranking(self):
        """Test CMC with a known imperfect ranking."""
        labels = np.array(['A', 'A', 'B', 'B'])
        # B0 now ranks A1 higher than B1 — incorrect rank-1
        scores = np.array([
            [0.0, 9.0, 2.0, 1.0],  # A0: rank-1 = A1 ✓
            [9.0, 0.0, 1.0, 2.0],  # A1: rank-1 = A0 ✓
            [1.0, 5.0, 0.0, 3.0],  # B0: rank-1 = A1 ✗, rank-2 = B1 ✓
            [2.0, 1.0, 8.0, 0.0],  # B1: rank-1 = B0 ✓
        ], dtype=np.float64)

        n = len(labels)
        rank1_correct = 0
        rank2_correct = 0
        for i in range(n):
            order = np.argsort(-scores[i])
            order_no_self = [idx for idx in order if idx != i]
            ordered_labels = labels[order_no_self]
            if labels[i] in ordered_labels[:1]:
                rank1_correct += 1
            if labels[i] in ordered_labels[:2]:
                rank2_correct += 1

        assert rank1_correct == 3  # 75% Rank-1
        assert rank2_correct == 4  # 100% Rank-2

    def test_map_perfect(self):
        """mAP should be 1.0 when all same-individual annotations are top-ranked."""
        scores, labels = self._make_score_matrix_and_labels()
        n = len(labels)

        ap_list = []
        for i in range(n):
            order = np.argsort(-scores[i])
            order_no_self = [idx for idx in order if idx != i]
            ordered_labels = labels[order_no_self]

            relevant = (ordered_labels == labels[i])
            if relevant.sum() == 0:
                continue
            cumsum = np.cumsum(relevant).astype(float)
            precision_at_k = cumsum / np.arange(1, len(relevant) + 1)
            ap = (precision_at_k * relevant).sum() / relevant.sum()
            ap_list.append(ap)

        mAP = np.mean(ap_list)
        assert mAP == 1.0
