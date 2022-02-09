"""Utility functions for Shared Interest."""

import numpy as np


def flatten(batch):
    """
    Flattens saliency by summing the channel dimension.

    Args:
    batch: 4D numpy array (batch, channels, height, width).

    Returns: 3D numpy array (batch, height, width) with the channel dimension
        summed.
    """
    return np.sum(batch, axis=1)


def normalize_0to1(batch):
    """
    Normalize a batch such that every value is in the range 0 to 1.

    Args:
    batch: a batch first numpy array to be normalized.

    Returns: A numpy array of the same size as batch, where each item in the
    batch has 0 <= value <= 1.
    """
    axis = tuple(range(1, len(batch.shape)))
    minimum = np.min(batch, axis=axis).reshape((-1,) + (1,) * len(axis))
    maximum = np.max(batch, axis=axis).reshape((-1,) + (1,) * len(axis))
    normalized_batch = (batch - minimum) / (maximum - minimum)
    return normalized_batch


def binarize_percentile(batch, percentile):
    """
    Creates binary mask by thresholding at percentile.

    Args:
    batch: 4D numpy array (batch, height, width).
    percentile: float in range 0 to 1. Values above the percentile value are 
        set to 1. Values below the percentile value are set to 0.

    Returns: A 4D numpy array with dtype uint8 with all values set to 0 or 1.
    """
    batch_size = batch.shape[0]
    batch_normalized = normalize_0to1(batch)
    percentile = np.percentile(batch_normalized, percentile * 100, axis=(1, 2)).reshape(batch_size, 1, 1)
    binary_mask = (batch_normalized >= percentile).astype('uint8')
    return binary_mask


def binarize_std(batch, num_std=1):
    """
    Creates binary mask by thresholding at num_std standard deviations above
    the mean.

    Args:
    batch: 3D numpy array (batch, height, width).
    num_std: int in range 0 to 3. Values above the (mean + num_std * std) value
        are set to 1. Values below are set to 0.

    Returns: A 3D numpy array with dtype uint8 with all values set to 0 or 1.
    """
    batch_size = batch.shape[0]
    batch_normalized = normalize_0to1(batch)
    mean = np.mean(batch_normalized, axis=(1, 2)).reshape(batch_size, 1, 1)
    std = np.std(batch_normalized, axis=(1, 2)).reshape(batch_size, 1, 1)
    threshold = mean + num_std * std
    binary_mask = (batch_normalized >= threshold).astype('uint8')
    return binary_mask