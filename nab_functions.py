import pandas as pd

from main_functions import calculate_loss, share_algorithm


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
    dt : pandas DataFrame
        The DataFrame of experts' predictions.
    
    """
    dt = pd.read_csv(
        f"NAB/results/{algorithm_list[0]}/{folder_name}/{algorithm_list[0]}{file_name}"
    )[["timestamp", "value", "label", "anomaly_score"]].rename(
        {"anomaly_score": f"score_{algorithm_list[0]}"}, axis=1
    )
    for algo_ind in algorithm_list[1:]:
        dt_temp = pd.read_csv(
            f"NAB/results/{algo_ind}/{folder_name}/{algo_ind}{file_name}"
        )[["anomaly_score"]].rename(
            {"anomaly_score": f"score_{algo_ind}"}, axis=1
        )
        dt = pd.merge(
            dt,
            dt_temp,
            how="left",
            left_index=True,
            right_index=True,
            validate="1:1",
        )
    return dt


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
    
    for m, share_type in enumerate(share_range):
        for n, alpha in enumerate(alpha_range):
            alpha0 = int(100 * alpha)
            score_share, loss_share, loss_experts, weights_experts = share_algorithm(
                target, score_experts, share_type, alpha
            )
            if (n == 0) & (m == 0):
                scores_share = pd.DataFrame(score_share)
                scores_share.columns = [f"score_{share_type}{alpha0}"]
            else:
                scores_share[f"score_{share_type}{alpha0}"] = score_share
    return scores_share