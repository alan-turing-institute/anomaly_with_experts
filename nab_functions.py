"""
Functions to create experts' and algorithms' predictions.
"""

import pandas as pd

from main_functions import share_algorithm


def read_nab(algorithm_list, folder_name, file_name):
    """
    The table of experts' predictions.

    Parameters
    ----------
    algorithm_list : list of str
        List of algorithms (experts).
    folder_name : str
    file_name : str

    Returns
    -------
    expert_predictions : pandas DataFrame
        The DataFrame of experts' predictions.
    """
    expert_predictions = pd.read_csv(
        f"NAB/results/{algorithm_list[0]}/{folder_name}/{algorithm_list[0]}{file_name}"
    )[["timestamp", "value", "label", "anomaly_score"]].rename(
        {"anomaly_score": f"score_{algorithm_list[0]}"}, axis=1
    )
    for algo_ind in algorithm_list[1:]:
        expert_predictions_add = pd.read_csv(
            f"NAB/results/{algo_ind}/{folder_name}/{algo_ind}{file_name}"
        )[["anomaly_score"]].rename({"anomaly_score": f"score_{algo_ind}"}, axis=1)
        expert_predictions = pd.merge(
            expert_predictions,
            expert_predictions_add,
            how="left",
            left_index=True,
            right_index=True,
            validate="1:1",
        )
    return expert_predictions


def get_scores(target, score_experts, share_range, alpha_range):
    """
    The table of the algorithms' predictions for a range of shares and alpha

    Parameters
    ----------
    target : numpy array
        Vector of binary outcomes (0 / 1), shape (number of steps, 1).
    score_experts : numpy array
        Array of experts' predictions, shape (number of steps, number of experts).
    share_range : list of str
        List of the algorithms ["Fixed", "Variable"].
    alpha : list of float
        List of the switching rates.

    Returns
    -------
    scores_share : pandas DataFrame
        The DataFrame of algorithms' predictions.
    """
    for m_share, share_type in enumerate(share_range):
        for n_alpha, alpha in enumerate(alpha_range):
            alpha0 = int(100 * alpha)
            score_share, *_ = share_algorithm(target, score_experts, share_type, alpha)
            if (n_alpha == 0) & (m_share == 0):
                scores_share = pd.DataFrame(score_share)
                scores_share.columns = [f"score_{share_type}{alpha0}"]
            else:
                scores_share[f"score_{share_type}{alpha0}"] = score_share
    return scores_share
