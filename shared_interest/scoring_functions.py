"""Scoring functions for Shared Interest."""

import numpy as np


def iou_coverage(ground_truth_features, saliency_features):
    """
    Returns the Shared Interest IoU Coverage metric. The result is computed as
        the number of features that occur in both the ground truth and saliency
        feature sets divided by the number of features that occur in either of
        the ground truth and saliency feature sets.

    Args:
    ground_truth_features: A binary array of size (batch_size, height, width)
        representing the ground truth features.
    saliency_features: A binary array of size (batch_size, height, width)
        representing the saliency features.
    """
    intersection = np.sum(ground_truth_features * saliency_features, axis=(1,2))
    union = np.sum(np.logical_or(ground_truth_features, saliency_features),
                   axis=(1,2))
    return intersection / union


def saliency_coverage(ground_truth_features, saliency_features):
    """
    Returns the Shared Interest Saliency Coverage metric. The result is computed
        as the saliency in the ground truth region divided by saliency
        everywhere.

    Args:
    ground_truth_features: A binary array of size (batch_size, height, width)
        representing the ground truth features.
    saliency_features: An array of size (batch_size, height, width)
        representing the saliency features. Array can be binary or continuous.
        If binary, the method computes the size of the intersection of the
        ground truth and saliency feature sets divided by the size of the
        saliency feature set. If continuous, the method computes the proportion
        of saliency within the ground truth region.
    """
    intersection = np.sum(ground_truth_features * saliency_features, axis=(1,2))
    explanation_saliency = np.sum(saliency_features, axis=(1,2))
    return intersection / explanation_saliency


def ground_truth_coverage(ground_truth_features, saliency_features):
    """
    Returns the Shared Interest Ground Truth Coverage metric. The result is
        computed as the number of features that occur in both of the ground
        truth and saliency feature sets divided by the number of ground truth
        features.

    Args:
    ground_truth_features: A binary array of size (batch_size, height, width)
        representing the ground truth features.
    saliency_features: A binary array of size (batch_size, height, width)
        representing the saliency features.
    """
    intersection = np.sum(ground_truth_features * saliency_features, axis=(1,2))
    ground_truth_saliency = np.sum(ground_truth_features, axis=(1,2))
    return intersection / ground_truth_saliency
