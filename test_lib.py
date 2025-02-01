from io import BytesIO

import pytest
import numpy as np

from aqua_blue import TimeSeries, utilities


def test_non_uniform_timestep_error():

    with pytest.raises(ValueError):
        _ = TimeSeries(dependent_variable=np.ones(10), times=np.logspace(0, 1, 10))


def test_zero_timestep_error():

    with pytest.raises(ValueError):
        _ = TimeSeries(dependent_variable=np.ones(10), times=np.zeros(10))


def test_can_save_and_load_time_series():

    t_original = TimeSeries(dependent_variable=np.ones(shape=(10, 2)), times=np.arange(10))
    with BytesIO() as buffer:
        t_original.save(buffer)
        buffer.seek(0)
        t_loaded = TimeSeries.from_csv(buffer)

    assert t_original == t_loaded


def test_normalizer_inversion():

    t_original = TimeSeries(dependent_variable=np.sin(np.arange(10)), times=np.arange(10))
    normalizer = utilities.Normalizer()
    t_normalized = normalizer.normalize(t_original)
    t_denormalized = normalizer.denormalize(t_normalized)

    assert t_original == t_denormalized
