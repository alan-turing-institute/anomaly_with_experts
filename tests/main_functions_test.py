import numpy as np
import pytest

import anomaly_delays.main_functions


@pytest.mark.parametrize(
    "share_type, expected_values",
    (
        pytest.param(
            "Fixed",
            [
                [1.00000005e-07, 2.23143551e-01],
                [6.93147181e-01, 1.00000005e-07],
                [1.20397280e00, 5.10825624e-01],
            ],
            id="Fixed share",
        ),
        pytest.param(
            "Variable",
            [[0.0, 0.04], [0.25, 0.0], [0.49, 0.16]],
            id="Variable share",
        ),
    ),
)
def test_calculate_loss(share_type, expected_values):
    target = np.array([[0], [1], [1]])
    score = np.array([[0, 0.2], [0.5, 1], [0.3, 0.6]])
    result = anomaly_delays.main_functions.calculate_loss(
        target, score, share_type=share_type
    )
    expected = np.array(expected_values)
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize(
    "share_type, expected_values",
    (
        pytest.param("Fixed", 0.58, id="Fixed share"),
        pytest.param("Variable", 0.5667982661056511, id="Variable share"),
    ),
)
def test_calculate_score(share_type, expected_values):
    score_experts = np.array([0.1, 0.9])
    weights_norm = np.array([0.4, 0.6])
    result = anomaly_delays.main_functions.calculate_score(
        score_experts, weights_norm, share_type=share_type
    )
    expected = expected_values
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "share_type, alpha, delay_current, expected_values",
    (
        pytest.param(
            "Fixed", 0.1, 5, [0.52397018, 0.47602982], id="Fixed share"
        ),
        pytest.param(
            "Variable", 0.05, 2, [0.6922423, 0.3077577], id="Variable share"
        ),
    ),
)
def test_calculate_weights(share_type, alpha, delay_current, expected_values):
    weights_norm = np.array([0.49, 0.51])
    loss_experts = np.array([0.1, 0.9])
    result = anomaly_delays.main_functions.calculate_weights(
        weights_norm,
        loss_experts,
        share_type=share_type,
        alpha=alpha,
        delay_current=delay_current,
    )
    expected = expected_values
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize(
    "share_type, alpha, delays, expected_score_share,"
    "expected_loss_share, expected_loss_experts, expected_weights_experts",
    (
        pytest.param(
            "Fixed",
            0.1,
            1,
            [0.5, 0.61],
            [0.69314718, 0.49429632],
            [[0.10536052, 2.30258509], [0.35667494, 1.60943791]],
            [[0.5, 0.5], [0.82, 0.18]],
            id="Fixed share",
        ),
        pytest.param(
            "Variable",
            0.3,
            1,
            [0.5, 0.63447463],
            [0.25, 0.13360879],
            [[0.01, 0.81], [0.09, 0.64]],
            [[0.5, 0.5], [0.87120567, 0.12879433]],
            id="Variable share",
        ),
    ),
)
def test_share_delays(
    share_type,
    alpha,
    delays,
    expected_score_share,
    expected_loss_share,
    expected_loss_experts,
    expected_weights_experts,
):
    target = np.array([[0], [1]])
    score_experts = np.array([[0.1, 0.9], [0.7, 0.2]])
    (
        result_score_share,
        result_loss_share,
        result_loss_experts,
        result_weights_experts,
    ) = anomaly_delays.main_functions.share_delays(
        target,
        score_experts,
        share_type=share_type,
        alpha=alpha,
        delays=delays,
    )
    np.testing.assert_array_almost_equal(
        expected_score_share, result_score_share
    )
    np.testing.assert_array_almost_equal(
        expected_loss_share, result_loss_share
    )
    np.testing.assert_array_almost_equal(
        expected_loss_experts, result_loss_experts
    )
    np.testing.assert_array_almost_equal(
        expected_weights_experts, result_weights_experts
    )
