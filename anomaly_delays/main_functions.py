"""
Main functions to calculate losses and predictions
of Fixed-share and Variable-share.
"""
import numpy as np
import pandas as pd

from anomaly_delays.helper_functions import create_delays_indices


def calculate_loss(target, score, share_type="Fixed", epsilon=1e-7):
    """
    Calculate the losses of predictions.

    Parameters
    ----------
    target : numpy array
        Vector of binary outcomes (0 / 1), shape (number of steps, 1).
    score : numpy array
        Array of predictions, shape (number of steps, number of predictors).
    share_type : {"Fixed", "Variable"}
    epsilon : float, optional
        Replace predictions 0 and 1 with epsilon and 1-epsilon
        for "Fixed" share_type (the default is 1e-7).

    Returns
    -------
    loss : numpy array
        Array of losses, shape (number of steps, number of predictors).


    Examples
    --------
    >>> target = np.array([[0], [1], [1]])
    >>> score = np.array([[0, 0.2], [0.5, 1], [0.3, 0.6]])
    >>> loss = calculate_loss(target, score, share_type="Variable")
    >>> loss
    array([[0.  , 0.04],
           [0.25, 0.  ],
           [0.49, 0.16]])
    """

    if share_type == "Fixed":
        score = score.clip(epsilon, 1 - epsilon)
        loss = -target * np.log(score) - (1 - target) * np.log(1 - score)
    else:
        assert share_type == "Variable"
        loss = (target - score) ** 2
    return loss


def calculate_score(
    score_experts, weights_norm, share_type="Fixed", epsilon=1e-7
):
    """
    Calculate the predictions of the algorithms at the current step.

    Parameters
    ----------
    score_experts : numpy array
        1D array of predictions, shape (1, number of experts).
    weights_norm : numpy array
        1D array of normalised experts' weights, shape (1, number of experts).
    share_type : {"Fixed", "Variable"}
    epsilon : float, optional
        Replace predictions 0 and 1 with epsilon and 1-epsilon
        for "Fixed" share_type (the default is 1e-7).

    Returns
    -------
    score_share : numpy.float64
        Prediction of the algorithm at the current step


    Examples
    --------
    >>> score_experts = np.array([0.1, 0.9])
    >>> weights_norm = np.array([0.4, 0.6])
    >>> score_share = calculate_score(score_experts, weights_norm, share_type="Fixed")
    >>> score_share
    0.58
    """
    assert len(score_experts) == len(weights_norm)
    if share_type == "Fixed":
        eta = 1
        score_experts = score_experts.clip(epsilon, 1 - epsilon)
        score_share = sum(weights_norm * score_experts)
    else:
        assert share_type == "Variable"
        eta = 2
        exp0 = np.exp(-eta * (score_experts) ** 2)
        exp1 = np.exp(-eta * (1 - score_experts) ** 2)
        generalised0 = -1 / eta * np.log(sum(weights_norm * exp0))
        generalised1 = -1 / eta * np.log(sum(weights_norm * exp1))
        score_share = 1 / 2 - (generalised1 - generalised0) / 2
    return score_share


def calculate_weights(
    weights_norm,
    loss_experts,
    share_type="Fixed",
    alpha=0,
    delay_current=1,
    epsilon=1e-7,
):
    """
    Calculate the experts' weights at the end of the step.

    Parameters
    ----------
    weights_norm : numpy array
        1D array of normalised experts' weights at the beginning of the step,
        shape (1, number of experts).
    loss_experts : numpy array
        1D array of losses summed over the delay length,
        shape (1, number of experts).
    share_type : {"Fixed", "Variable"}
    alpha : float
        The switching rate between experts
        (the default is 0, i.e. the Aggregating Algorithm).
    delay_current : int
        The current delay length (the default is 1, i.e. feedback comes
        immediately after making the prediction).
    epsilon : float, optional
        Replace weights below epsilon with epsilon (the default is 1e-7).

    Returns
    -------
    weights_norm_new : numpy array
        1D array of normalised experts' weights at the end of the step,
        shape (1, number of experts).


    Examples
    --------
    >>> weights_norm = np.array([0.49, 0.51])
    >>> loss_experts = np.array([0.1, 0.9])
    >>> weights_norm_new = calculate_weights(
    ...     loss_experts,
    ...     weights_norm,
    ...     share_type="Fixed",
    ...     alpha=0.1,
    ...     delay_current=5,
    ... )
    >>> weights_norm_new
    array([0.52397018, 0.47602982])
    """

    assert len(loss_experts) == len(weights_norm)
    experts_number = len(loss_experts)
    if share_type == "Fixed":
        eta = 1
        weights_tilde = weights_norm * np.exp(
            -eta * loss_experts / delay_current
        )
        pool = alpha * np.sum(weights_tilde)
        weights_update = (1 - alpha) * weights_tilde + (
            pool - alpha * weights_tilde
        ) / (experts_number - 1)
    else:
        assert share_type == "Variable"
        eta = 2
        weights_tilde = weights_norm * np.exp(
            -eta * loss_experts / delay_current
        )
        pool = np.sum((1 - (1 - alpha) ** loss_experts) * weights_tilde)
        weights_update = (1 - alpha) ** loss_experts * weights_tilde + (
            pool - (1 - (1 - alpha) ** loss_experts) * weights_tilde
        ) / (experts_number - 1)
    weights_norm_new = weights_update / np.sum(weights_update)
    weights_norm_new = weights_norm_new.clip(epsilon)
    weights_norm_new = weights_norm_new / np.sum(weights_norm_new)
    return weights_norm_new


def share_delays(
    target, score_experts, share_type="Fixed", alpha=0, delays=1, epsilon=1e-7
):
    """
    Calculate the predictions and losses of Fixed-share and Variable-share,
    the losses and weights of the experts with the delayed feedback.

    Parameters
    ----------
    target : numpy array
        Vector of binary outcomes (0 / 1), shape (number of steps, 1).
    score_experts : numpy array
        Array of experts' predictions,
        shape (number of steps, number of experts).
    share_type : {"Fixed", "Variable"}
        Type of the algorithm (the default is Fixed-share).
    alpha : float
        The switching rate between experts
        (the default is 0, i.e. the Aggregating Algorithm).
    delays : integer or sequence of integers
        Subsequent delays lengths (the default is 1, i.e. feedback comes
        immediately after making the prediction).
    epsilon : float, optional
        Replace predictions 0 and 1 with epsilon and 1-epsilon
        for "log" loss_type (the default is 1e-7).

    Returns
    -------
    score_share : numpy array
        The algorithm's predictions, shape (number of steps, 1).
    loss_share : numpy array
        Array of the algorithm's losses, shape (number of steps, 1).
    loss_experts : numpy array
        Array of experts' losses, shape (number of steps, number of experts).
    weights_norm : numpy array
        Array of experts' normalised weights
        (number of steps, number of experts).


    Examples
    --------
    >>> target = np.array([[0], [1]])
    >>> score_experts = np.array([[0.1, 0.9], [0.7, 0.2]])
    >>> score_share, loss_share, loss_experts, weights_experts = share_delays(
    ...     target, score_experts, share_type="Variable", alpha=0.3
    ... )
    >>> score_share
    array([0.5       , 0.63447463])
    >>> loss_share
    array([0.25      , 0.13360879])
    >>> loss_experts
    array([[0.01, 0.81],
           [0.09, 0.64]])
    >>> weights_experts
    array([[0.5       , 0.5       ],
           [0.87120567, 0.12879433]])
    """

    steps_number, experts_number = score_experts.shape
    loss_experts = np.zeros((steps_number, experts_number))
    score_share = np.zeros(steps_number)
    loss_share = np.zeros(steps_number)
    weights_norm = np.zeros((steps_number + 1, experts_number))
    weights_norm[0] = np.repeat(1, experts_number) / experts_number
    delays_indices_array = create_delays_indices(steps_number, delays)
    for i in range(steps_number):
        score_share[i] = calculate_score(
            score_experts[i], weights_norm[i], share_type
        )
        loss_experts[i] = calculate_loss(
            target[i], score_experts[i], share_type
        )
        loss_share[i] = calculate_loss(target[i], score_share[i], share_type)
        if any(delays_indices_array == i):
            delays_index = delays_indices_array.tolist().index(i)
            if delays_index == 0:
                delay_current = delays_indices_array[0] + 1
            else:
                delay_current = i - delays_indices_array[delays_index - 1]
            loss_experts_sum = np.sum(
                loss_experts[
                    (i - delay_current + 1) : (i + 1),
                ],
                axis=0,
            )
            weights_norm[i + 1] = calculate_weights(
                weights_norm[i],
                loss_experts_sum,
                share_type,
                alpha,
                delay_current,
                epsilon,
            )
        else:
            weights_norm[i + 1] = weights_norm[i]
    return score_share, loss_share, loss_experts, weights_norm[:-1]


def get_scores(
    target, score_experts, share_range, alpha_range, delays_range=(1,)
):
    """
    Apply the algorithms under delayed feedback
    for a range of shares, alpha, and delays.

    Parameters
    ----------
    target : numpy array
        Vector of binary outcomes (0 / 1), shape (number of steps, 1).
    score_experts : numpy array
        Array of experts' predictions,
        shape (number of steps, number of experts).
    share_range : tuple of str
        Tuple of the algorithms ["Fixed", "Variable"].
    alpha : tuple of float
        Tuple of the switching rates.
    delays_range : tuple of int, arrays, lists
        Tuple of the delays.

    Returns
    -------
    scores_share : pandas DataFrame
        The DataFrame of algorithms' predictions.
    """
    for m_share, share_type in enumerate(share_range):
        for n_alpha, alpha in enumerate(alpha_range):
            for l_delays, delays in enumerate(delays_range):
                alpha0 = int(100 * alpha)
                score_share, *_ = share_delays(
                    target, score_experts, share_type, alpha, delays
                )
                if (n_alpha == 0) & (m_share == 0) & (l_delays == 0):
                    scores_share = pd.DataFrame(score_share)
                    scores_share.columns = [
                        f"score_{share_type}{alpha0}d{l_delays}"
                    ]
                else:
                    scores_share[
                        f"score_{share_type}{alpha0}d{l_delays}"
                    ] = score_share
    return scores_share
