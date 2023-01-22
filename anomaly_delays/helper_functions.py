"""
Helper functions to read data, create delays,
calculate cumulative average losses.
"""
import os

import numpy as np
import pandas as pd


def read_nab(algorithm_list, folder_name, file_name):
    """
    Read the table of experts' predictions.

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
        os.path.join(
            "NAB/results",
            f"{algorithm_list[0]}/{folder_name}/{algorithm_list[0]}{file_name}",
        )
    )[["timestamp", "value", "label", "anomaly_score"]].rename(
        {"anomaly_score": f"score_{algorithm_list[0]}"}, axis=1
    )
    for algo_ind in algorithm_list[1:]:
        expert_predictions_add = pd.read_csv(
            os.path.join(
                "NAB/results",
                f"{algo_ind}/{folder_name}/{algo_ind}{file_name}",
            )
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


def create_delays_indices(steps_number, delays=1):
    """
    Create the array of indices when the feedback is received.

    The helper function which creates an array of indices based on a sequence
    of integers or a single integer, which represents the length of the delay.
    For example, sequence of delays [2, 5] means that we receive the first
    feedback after two steps, and the second - after another five steps.
    Here we also use the number of steps to make the array, though we do not
    need it for making predictions.

    Parameters
    ----------
    steps_number : integer
        The total number of steps.
    delays : integer or sequence of integers
        Subsequent delays lengths (the default is 1, i.e. feedback comes
        immediately after making the prediction).

    Returns
    -------
    delays_array : numpy array
        1D array of indices when the feedback is received.


    Examples
    --------
    >>> create_delays_indices(steps_number=7, delays=3)
    array([2, 5])
    >>> create_delays_indices(steps_number=9, delays=[2, 5])
    array([1, 6])
    >>> create_delays_indices(steps_number=9, delays=np.array([2, 5]))
    array([1, 6])
    """
    if isinstance(delays, int):
        delays_array = np.cumsum(np.repeat(delays, np.floor(steps_number / delays)))
    else:
        assert sum(delays) <= steps_number
        delays_array = np.cumsum(delays)
    return delays_array - 1


def generate_random_delays(max_length, min_delay, max_delay):
    """
    Generate random delays list, between min_delay and max_delay,
    with total number of delays less or equal to max_length.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> generate_random_delays(max_length=10, min_delay=1, max_delay=4)
    [3, 4, 1]
    """
    random_delay_list = []
    while sum(random_delay_list) <= max_length:
        random_delay = np.random.randint(min_delay, max_delay + 1)
        if sum(random_delay_list) + random_delay < max_length:
            random_delay_list.append(random_delay)
        else:
            break
    return random_delay_list


def calc_avg_loss(losses, current_delay, flag_cumulative=False):
    """
    Calculate cumulative average losses
    for the array of losses and the current delay.
    """
    delays_array = create_delays_indices(
        steps_number=losses.shape[0], delays=current_delay
    )
    delays_group = np.repeat(0, delays_array[-1] + 1)
    for j in range(delays_array.shape[0]):
        if j < delays_array.shape[0] - 1:
            delays_group[(delays_array[j] + 1) : (delays_array[j + 1] + 1)] = j + 1
    losses_trim = losses[: (delays_array[-1] + 1)].copy()
    losses_trim["delay_group"] = delays_group
    losses_avg = losses_trim.groupby(["delay_group"], as_index=False).mean()
    losses_avg_cumsum = losses_avg.cumsum()
    losses_avg_cumsum["delay_group"] = losses_avg_cumsum.index
    if flag_cumulative:
        losses_return = losses_avg_cumsum
    else:
        losses_return = losses_avg
    return losses_return
