"""
Main functions to calculate losses and predictions of Fixed-share and Variable-share.
"""
import numpy as np


def calculate_loss(target, score, loss_type="log", epsilon=1e-7):
    """

    The losses of predictions.

    Parameters
    ----------
    target : numpy array
        Vector of binary outcomes (0 / 1), shape (number of steps, 1).
    score : numpy array
        Array of predictions, shape (number of steps, number of predictors).
    loss_type : {"log", "square"}
    epsilon : float, optional
        Replace predictions 0 and 1 with epsilon and 1-epsilon
        for "log" loss_type (the default is 1e-7).

    Returns
    -------
    loss : numpy array
        Array of losses, shape (number of steps, number of predictors).


    Examples
    --------
    >>> target = np.array([[0], [1], [1]])
    >>> score = np.array([[0, 0.2], [0.5, 1], [0.3, 0.6]])
    >>> loss = calculate_loss(target, score, loss_type = "square")
    >>> loss
    array([[0.  , 0.04],
           [0.25, 0.  ],
           [0.49, 0.16]])
    """

    if loss_type == "log":
        score = score.clip(epsilon, 1 - epsilon)
        loss = -target * np.log(score) - (1 - target) * np.log(1 - score)
    else:
        assert loss_type == "square"
        loss = (target - score) ** 2
    return loss


def share_algorithm(target, score_experts, share_type="Fixed", alpha=0, epsilon=1e-7):
    """

    The implementation of Fixed-share and Variable-share.

    Parameters
    ----------
    target : numpy array
        Vector of binary outcomes (0 / 1), shape (number of steps, 1).
    score_experts : numpy array
        Array of experts' predictions, shape (number of steps, number of experts).
    share_type : {"Fixed", "Variable"}
        Type of the algorithm (the default is Fixed-share).
    alpha : float
        The switching rate between experts (the default is 0, i.e. the Aggregating Algorithm).
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
        Array of experts' normalised weights (number of steps, number of experts).


    Examples
    --------
    >>> target = np.array([[0], [1]])
    >>> score = np.array([[0.1, 0.9], [0.7, 0.2]])
    >>> score_share, loss_share, loss_experts, weights_experts = share_algorithm(
    ...     target,
    ...     score,
    ...     share_type = "Variable",
    ...     alpha = 0.3
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

    [steps_number, experts_number] = score_experts.shape
    initial_weights = np.repeat(1, experts_number)
    loss_experts = np.zeros((steps_number, experts_number))
    weights_norm = np.zeros((steps_number + 1, experts_number))
    score_share = np.zeros(steps_number)
    loss_share = np.zeros(steps_number)
    weights_norm[0] = initial_weights / sum(initial_weights)
    for i in range(steps_number):
        if share_type == "Fixed":
            loss_type = "log"
            eta = 1
            score_experts[i] = score_experts[i].clip(epsilon, 1 - epsilon)
            score_share[i] = sum(weights_norm[i] * score_experts[i])
        else:
            assert share_type == "Variable"
            loss_type = "square"
            eta = 2
            exp0 = np.exp(-eta * (score_experts[i]) ** 2)
            exp1 = np.exp(-eta * (1 - score_experts[i]) ** 2)
            generalised0 = -1 / eta * np.log(sum(weights_norm[i] * exp0))
            generalised1 = -1 / eta * np.log(sum(weights_norm[i] * exp1))
            score_share[i] = 1 / 2 - (generalised1 - generalised0) / 2
        loss_experts[i] = calculate_loss(target[i], score_experts[i], loss_type)
        loss_share[i] = calculate_loss(target[i], score_share[i], loss_type)
        weights_tilde = weights_norm[i] * np.exp(-eta * loss_experts[i])
        if share_type == "Fixed":
            pool = alpha * np.sum(weights_tilde)
            weights_update = (1 - alpha) * weights_tilde + (
                pool - alpha * weights_tilde
            ) / (experts_number - 1)
        else:
            assert share_type == "Variable"
            pool = np.sum((1 - (1 - alpha) ** loss_experts[i]) * weights_tilde)
            weights_update = (1 - alpha) ** loss_experts[i] * weights_tilde + (
                pool - (1 - (1 - alpha) ** loss_experts[i]) * weights_tilde
            ) / (experts_number - 1)
        weights_norm[i + 1] = weights_update / np.sum(weights_update)
    return score_share, loss_share, loss_experts, weights_norm[:-1]
