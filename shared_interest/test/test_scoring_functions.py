"""Tests for scoring functions."""

import unittest
import numpy as np

from shared_interest.scoring_functions import iou_coverage, saliency_coverage, ground_truth_coverage


class TestIoUCoverage(unittest.TestCase):
    """Tests for IoU Coverage."""

    def setUp(self):
        self.shape = (5, 250, 250)
        self.ground_truth_features = np.zeros(self.shape)
        self.ground_truth_features[0, 0:100, 0:100] = 1
        self.ground_truth_features[1, 100:250, 200:250] = 1
        self.ground_truth_features[2, 0:250, 0:250] = 1
        self.ground_truth_features[3, 0:1, 0:1] = 1
        self.ground_truth_features[4, 50:100, 50:100] = 1

    def test_iou_coverage(self):
        """Tests iou_coverage."""
        saliency_features = np.zeros(self.shape)
        saliency_features[0, 50:150, 50:150] = 1    # overlapping regions
        saliency_features[1, 50:250, 100:250] = 1   # ground truth in saliency
        saliency_features[2, 50:100, 50:100] = 1    # saliency in ground truth
        saliency_features[3, 100:150, 150:250] = 1  # no overlap
        saliency_features[4, 50:100, 50:100] = 1    # identical regions

        scores = iou_coverage(self.ground_truth_features, saliency_features)
        self.assertTupleEqual(scores.shape, (self.shape[0],))
        self.assertIs(type(scores), np.ndarray)

        expected_scores = np.array([0.1428571429, 0.25, 0.04, 0.0, 1.0])
        self.assertTrue(
            np.allclose(scores, expected_scores),
            'Expected: %s got: %s' %(str(expected_scores), str(scores))
        )


class TestSaliencyCoverage(unittest.TestCase):
    """Tests for Saliency Coverage."""

    def setUp(self):
        self.shape = (5, 250, 250)
        self.ground_truth_features = np.zeros(self.shape)
        self.ground_truth_features[0, 0:100, 0:100] = 1
        self.ground_truth_features[1, 100:250, 200:250] = 1
        self.ground_truth_features[2, 0:250, 0:250] = 1
        self.ground_truth_features[3, 0:1, 0:1] = 1
        self.ground_truth_features[4, 50:100, 50:100] = 1

    def test_binary_saliency_coverage(self):
        """Tests binary ground truth and saliency regions."""
        saliency_features = np.zeros(self.shape)
        saliency_features[0, 50:150, 50:150] = 1    # overlapping regions
        saliency_features[1, 50:250, 100:250] = 1   # ground truth in saliency
        saliency_features[2, 50:100, 50:100] = 1    # saliency in ground truth
        saliency_features[3, 100:150, 150:250] = 1  # no overlap
        saliency_features[4, 50:100, 50:100] = 1    # identical regions

        scores = saliency_coverage(self.ground_truth_features,
                                   saliency_features)
        self.assertTupleEqual(scores.shape, (self.shape[0],))
        self.assertIs(type(scores), np.ndarray)

        expected_scores = np.array([0.25, 0.25, 1.0, 0.0, 1.0])
        self.assertTrue(
            (scores == expected_scores).all(),
            'Expected: %s got: %s' %(str(expected_scores), str(scores)))


    def test_continuous_saliency_coverage(self):
        """Tests continuous ground truth and saliency regions."""
        saliency_features = np.zeros(self.shape)
        saliency_features[0, 50:150, 50:150] = 0.2    # overlapping regions
        saliency_features[1, 50:250, 100:250] = 0.5   # ground truth in saliency
        saliency_features[2, 50:100, 50:100] = 0.9    # saliency in ground truth
        saliency_features[3, 100:150, 150:250] = 0.1  # no overlap
        saliency_features[4, 50:100, 50:100] = 0.6    # identical regions

        scores = saliency_coverage(self.ground_truth_features,
                                   saliency_features)
        self.assertTupleEqual(scores.shape, (self.shape[0],))
        self.assertIs(type(scores), np.ndarray)

        expected_scores = np.array([0.25, 0.25, 1.0, 0.0, 1.0])
        self.assertTrue(
            np.allclose(scores, expected_scores),
            'Expected: %s got: %s' %(str(expected_scores), str(scores)))


class TestGroundTruthCoverage(unittest.TestCase):
    """Tests for Ground Truth Coverage."""

    def setUp(self):
        self.shape = (5, 250, 250)
        self.ground_truth_features = np.zeros(self.shape)
        self.ground_truth_features[0, 0:100, 0:100] = 1
        self.ground_truth_features[1, 100:250, 200:250] = 1
        self.ground_truth_features[2, 0:250, 0:250] = 1
        self.ground_truth_features[3, 0:1, 0:1] = 1
        self.ground_truth_features[4, 50:100, 50:100] = 1

    def test_ground_truth_coverage(self):
        """Tests ground_truth_coverage."""
        saliency_features = np.zeros(self.shape)
        saliency_features[0, 50:150, 50:150] = 1    # overlapping regions
        saliency_features[1, 50:250, 100:250] = 1   # ground truth in saliency
        saliency_features[2, 50:100, 50:100] = 1    # saliency in ground truth
        saliency_features[3, 100:150, 150:250] = 1  # no overlap
        saliency_features[4, 50:100, 50:100] = 1    # identical regions

        scores = ground_truth_coverage(self.ground_truth_features,
                                       saliency_features)
        self.assertTupleEqual(scores.shape, (self.shape[0],))
        self.assertIs(type(scores), np.ndarray)

        expected_scores = np.array([0.25, 1.0, 0.04, 0.0, 1.0])
        self.assertTrue(
            np.allclose(scores, expected_scores),
            'Expected: %s got: %s' %(str(expected_scores), str(scores)))



if __name__ == '__main__':
    unittest.main()
