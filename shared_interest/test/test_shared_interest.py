"""Tests for Shared Interest."""

import unittest
import numpy as np

from shared_interest.shared_interest import shared_interest


class TestSharedInterest(unittest.TestCase):
    """Tests for Shared Interest."""

    def setUp(self):
        self.shape = (5, 250, 250)

        self.ground_truth_features = np.zeros(self.shape).astype(int)
        self.ground_truth_features[0, 0:100, 0:100] = 1
        self.ground_truth_features[1, 100:250, 200:250] = 1
        self.ground_truth_features[2, 0:250, 0:250] = 1
        self.ground_truth_features[3, 0:1, 0:1] = 1
        self.ground_truth_features[4, 50:100, 50:100] = 1

        self.binary_saliency_features = np.zeros(self.shape).astype(int)
        self.binary_saliency_features[0, 50:150, 50:150] = 1    # overlapping regions
        self.binary_saliency_features[1, 50:250, 100:250] = 1   # ground truth in saliency
        self.binary_saliency_features[2, 50:100, 50:100] = 1    # saliency in ground truth
        self.binary_saliency_features[3, 100:150, 150:250] = 1  # no overlap
        self.binary_saliency_features[4, 50:100, 50:100] = 1    # identical regions

        self.continuous_saliency_features = np.zeros(self.shape)
        self.continuous_saliency_features[0, 50:150, 50:150] = 0.2   # overlapping regions
        self.continuous_saliency_features[1, 50:250, 100:250] = 0.5  # ground truth in saliency
        self.continuous_saliency_features[2, 50:100, 50:100] = 0.9   # saliency in ground truth
        self.continuous_saliency_features[3, 100:150, 150:250] = 0.1 # no overlap
        self.continuous_saliency_features[4, 50:100, 50:100] = 0.6   # identical regions


    def test_shared_interest_valid_input(self):
        """Test shared interest on valid inputs."""
        # Test IoU Coverage.
        iou_shared_interest = shared_interest(self.ground_truth_features,
                                              self.binary_saliency_features,
                                              score='iou_coverage')
        self.assertIs(type(iou_shared_interest), np.ndarray)
        self.assertTupleEqual(iou_shared_interest.shape, (self.shape[0],))
        expected_iou_scores = np.array([0.1428571429, 0.25, 0.04, 0.0, 1.0])
        self.assertTrue(np.allclose(iou_shared_interest, expected_iou_scores),
                        'Expected: %s got: %s' %(str(expected_iou_scores),
                                                 str(iou_shared_interest)))

        # Test Binary Saliency Coverage.
        saliency_coverage_binary_shared_interest = shared_interest(
            self.ground_truth_features,
            self.binary_saliency_features,
            score='saliency_coverage')
        self.assertIs(type(saliency_coverage_binary_shared_interest),
                      np.ndarray)
        self.assertTupleEqual(saliency_coverage_binary_shared_interest.shape,
                              (self.shape[0],)
                             )
        expected_saliency_coverage_scores = np.array([0.25, 0.25, 1.0, 0.0, 1.0])
        self.assertTrue(
            np.allclose(saliency_coverage_binary_shared_interest,
                        expected_saliency_coverage_scores
                       ),
            'Expected: %s got: %s' %(
                str(expected_saliency_coverage_scores),
                str(saliency_coverage_binary_shared_interest)
            )
        )

        # Test Continuous Saliency Coverage.
        saliency_coverage_continuous_shared_interest = shared_interest(
            self.ground_truth_features,
            self.continuous_saliency_features,
            score='saliency_coverage'
        )
        self.assertIs(type(saliency_coverage_continuous_shared_interest),
                      np.ndarray)
        self.assertTupleEqual(
            saliency_coverage_continuous_shared_interest.shape,
            (self.shape[0],)
        )
        self.assertTrue(
            np.allclose(saliency_coverage_continuous_shared_interest,
                        expected_saliency_coverage_scores),
            'Expected: %s got: %s' %(
                str(expected_saliency_coverage_scores),
                str(saliency_coverage_continuous_shared_interest)
            )
        )

        # Test Ground Truth Coverage.
        ground_truth_coverage_shared_interest = shared_interest(
            self.ground_truth_features,
            self.binary_saliency_features,
            score='ground_truth_coverage')
        self.assertIs(type(ground_truth_coverage_shared_interest), np.ndarray)
        self.assertTupleEqual(ground_truth_coverage_shared_interest.shape,
                              (self.shape[0],)
                             )
        expected_ground_truth_coverage_scores = np.array([0.25, 1.0, 0.04,
                                                          0.0, 1.0])
        self.assertTrue(
            np.allclose(ground_truth_coverage_shared_interest,
                        expected_ground_truth_coverage_scores
                       ),
            'Expected: %s got: %s' %(
                str(expected_ground_truth_coverage_scores),
                str(ground_truth_coverage_shared_interest)))


    def test_shared_interest_invalid_inputs(self):
        """Tests shared interest on invalid inputs."""
        # Invalid score function
        with self.assertRaises(ValueError):
            shared_interest(self.ground_truth_features,
                            self.binary_saliency_features,
                            score='score')

        # Continuous input with non-continuous scoring function
        with self.assertRaises(ValueError):
            shared_interest(self.ground_truth_features,
                            self.continuous_saliency_features,
                            score='iou_coverage')
        with self.assertRaises(ValueError):
            shared_interest(self.ground_truth_features,
                            self.continuous_saliency_features,
                            score='ground_truth_coverage')

        # Binary input with non-binary values
        with self.assertRaises(ValueError):
            bad_saliency_region = self.binary_saliency_features.copy()
            bad_saliency_region[:, 0:100, 30:90] = 2
            shared_interest(self.ground_truth_features,
                            bad_saliency_region,
                            score='iou_coverage')

        # Non-binary ground truth features
        with self.assertRaises(ValueError):
            shared_interest(self.continuous_saliency_features,
                            self.binary_saliency_features)


if __name__ == '__main__':
    unittest.main()
