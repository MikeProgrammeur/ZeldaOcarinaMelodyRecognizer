import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

DEFAULT_LOSS = "rmse" 
DEFAULT_HUBER_DELTA = 1

def match_sequence(measured, target, loss=DEFAULT_LOSS, huber_delta=DEFAULT_HUBER_DELTA):
    """Try to identify the sequence 'target' inside of a longer sequence 'measured', return the loss between target and the optimal subsequence of measured"""
    measured = np.asarray(measured, dtype=float)
    target = np.asarray(target, dtype=float)

    if measured.ndim != 1 or target.ndim != 1:
        raise ValueError("Both sequences must be 1D")
    if len(target) == 0 or len(measured) < len(target): # no melody can be identified so return infinite loss
        return np.inf
    if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(target)):
        raise ValueError("Sequences must contain only finite values")

    windows = sliding_window_view(measured, len(target))
    errors = windows - target

    match loss:
        case "mae":
            scores = np.mean(np.abs(errors), axis=1)

        case "rmse":
            scores = np.sqrt(np.mean(errors**2, axis=1))

        case "max":
            scores = np.max(np.abs(errors), axis=1)

        case "huber":
            abs_errors = np.abs(errors)
            huber = np.where(
                abs_errors <= huber_delta,
                0.5 * errors**2,
                huber_delta * (abs_errors - 0.5 * huber_delta)
            )
            scores = np.mean(huber, axis=1)

        case _:
            raise ValueError(f"Unknown loss: {loss}")

    best_idx = np.argmin(scores)

    return float(scores[best_idx])

def identify_melody(measured, target_dic, detection_threshold, loss=DEFAULT_LOSS, huber_delta=DEFAULT_HUBER_DELTA):
    """Use match_sequence on a sequence and try every value of a dictionary, return the key corresponding to the minimal loss"""
    min_error = np.inf
    res = None

    for key, value in target_dic.items():
        current_error = match_sequence(
            measured, value, loss=loss, huber_delta=huber_delta
        )

        if current_error < min_error:
            min_error = current_error
            res = key if min_error < detection_threshold else None

    return res
                