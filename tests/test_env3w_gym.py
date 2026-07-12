import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from classes._Env3WGym import Env3WGym


def test_env_step_normalizes_array_action() -> None:
    dataset = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0],
            [0.2, 0.3, 0.4, 0.5, 0.6, 1],
        ],
        dtype=float,
    )
    env = Env3WGym(dataset)

    obs, reward, done, info = env.step(np.array([0]))

    assert obs.shape == (5,)
    assert reward == 0.0
    assert done is False
    assert info["action"] == 0


def test_env_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError):
        Env3WGym(np.array([]))
