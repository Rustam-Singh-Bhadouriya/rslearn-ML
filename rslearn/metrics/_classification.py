"""
_classification.py  

This File Contains Many Important tools and sets of algorithams for classification tasks like  

- `Accuracy Score (accuracy_score)`  
- `log loss (coming soon)`  
- `classification report (coming soon)`  
- `precesion score (coming soon)`  
- `recall score (coming soon)`  

"""

import numpy as np

def accuracy_score(y_true, y_pred, weights=None, multi_output="uniform_average", normalize=True):
    """Accuracy classification score.

    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.

    Read more in the `Documentation or README.md`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels. Sparse matrix is only supported when
        labels are of :term:`multilabel` type.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier. Sparse matrix is only
        supported when labels are of :term:`multilabel` type.

    normalize : bool, default=True
        If ``False``, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.

    weights : array-like of shape (n_samples,), default=None
        Sample weights.
    
    multi_output : {"uniform_average", "weighted", "raw_values"}, default="uniform_average"
        Defines how to aggregate scores for multi-output data:
        
        - "uniform_average" : Average scores across all outputs (default)
        - "weighted"        : Weighted average using `weights`
        - "raw_values"      : Return scores for each output separately

    Returns
    -------
    score : float
        If ``normalize == True``, returns the fraction of correctly classified samples,
        else returns the number of correctly classified samples.

        The best performance is 1.0 with ``normalize == True`` and the number
        of samples with ``normalize == False``.


    Examples
    --------
    >>> from rslearn.metrics import accuracy_score
    >>> y_pred = [0, 2, 1, 3]
    >>> y_true = [0, 1, 2, 3]
    >>> accuracy_score(y_true, y_pred)
    0.5
    >>> accuracy_score(y_true, y_pred, normalize=False)
    2.0

    In the multilabel case with binary label indicators:

    >>> import numpy as np
    >>> accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))
    0.75
    """

    valid_params = {"uniform_average", "raw_values", "weighted"}
    if multi_output not in valid_params:
        raise ValueError(
            f"Got Invalid Parameter, {multi_output}. But supported {valid_params} only."
        )

    # Converting to Arrays
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # Checking for Array Size MisMatch
    if len(y_true) != len(y_pred):
        raise ValueError(f"Array Size Mismatch {(len(y_true, y_pred))}")
    
    
    # Handling Single Output Metrics
    if y_true.ndim == 1:
        return _accuracy_score_helper_1d(y_true=y_true, y_pred=y_pred, normalize=normalize)
    
    if y_true.shape[1] == 1:
        return _accuracy_score_helper_1d(y_true=y_true, y_pred=y_pred, normalize=normalize)
    
    outputs = _accuracy_score_helper_2d(y_true=y_true, y_pred=y_pred, normalize=normalize)

    if multi_output == "uniform_average":
        return np.mean(outputs)
    
    if multi_output == "raw_values":
        return outputs
    
    if multi_output == "weighted":
        if weights is None:
            raise ValueError(f"weights aren't given, Enter weights in function parameter")
        
        weights = np.asarray(weights, dtype=float)

        if weights.shape[0] != len(outputs):
            raise ValueError(f"Invalid Weight Size got {weights.shape[0]}, needed {len(outputs)}")
        
        np.average(outputs, weights=weights)

    

        
"""Accuracy score Helper for 1D"""
def _accuracy_score_helper_1d(y_true, y_pred, normalize=True):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    
    correct_count = 0

    length = len(y_true)

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct_count += 1

    # Returning Accuracy
    if normalize is False:
        return correct_count
    
    return correct_count/length

"""Accuracy score Helper for 2D sparse metrics"""
def _accuracy_score_helper_2d(y_true, y_pred, normalize=True):
    n_col = y_true.shape[1]
    
    output = []
    for col in range(n_col):
        selected_true = y_true[:, col]
        selected_pred = y_pred[:, col]

        accuracy = _accuracy_score_helper_1d(selected_true, selected_pred, normalize=normalize)
        output.append(accuracy)
    
    return np.array(output)