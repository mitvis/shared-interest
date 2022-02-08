"""Shared Interest method"""

import inspect
import numpy as np

from shared_interest import scoring_functions


def shared_interest(ground_truth_features, saliency_features,
                    score='iou_coverage'):
    """
    Returns the Shared Interest score for the given ground truth and saliency
        features.

    Args:
    ground_truth_features: A binay array of size (batch_size, height, width)
        representing the ground truth features. 1 represents features in the
        ground truth and 0 represents features not in the ground truth.
    saliency_features: An array of size (batch_size, height, width) representing
        the saliency features. If the array is binary (contains only 0s and 1s),
        set-based scoring is used and all scoring functions will work. If the
        array is continuous, only saliency_coverage scoring can be used and the
        proportion of saliency in the ground truth region will be returned.
    score: One of the strings: 'iou_coverage', 'ground_truth_coverage', or
        'saliency_coverage' indicating which scoring function to use.

    Raises:
        ValueError if score is not a valid scoring function.
        ValueError if saliency_features is no binary (contains values other
            than 0 or 1) and the score is not 'saliency_coverage'.
        ValueError if ground_truth_features is not binary.

    Returns:
    A numpy array of size (batch_size) of floating point shared interest scores.
    """
    ground_truth_features = _convert_to_numpy(ground_truth_features)
    saliency_features = _convert_to_numpy(saliency_features)

    # Check input invariances.
    if not _is_binary(ground_truth_features):
        raise ValueError('ground_truth_features must be binary array.')
    if ground_truth_features.shape != saliency_features.shape:
        raise ValueError('ground_truth_features and saliency_features must \
                         be the same shape.')
    if not _is_binary(saliency_features) and score != 'saliency_coverage':
        raise ValueError('Non-binary saliency features can only use \
                         saliency_coverage score.')

    # Get the scoring function from score input.
    score_functions = dict(inspect.getmembers(scoring_functions,
                                              inspect.isfunction)
                          )
    if score not in score_functions:
        raise ValueError('%s is not a valid scoring function.' %(score))
    score_function = score_functions[score]

    # Compute the shared interest score.
    score = score_function(ground_truth_features, np.abs(saliency_features))
    return score


def _is_binary(array):
    """Checks if array only contains 0s and 1s."""
    return np.isin(array, [0, 1]).all()


def _convert_to_numpy(array):
    """Converys array to a numpy array if it is not already a numpy array."""
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    return array
        