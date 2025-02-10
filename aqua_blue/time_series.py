from typing import IO, Union
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class TimeSeries:
    dependent_variable: NDArray
    times: NDArray

    def __post_init__(self):
        timesteps = np.diff(self.times)
        if not np.isclose(np.std(timesteps), 0.0):
            raise ValueError("TimeSeries.times must be uniformly spaced")
        if np.isclose(np.mean(timesteps), 0.0):
            raise ValueError("TimeSeries.times must have a timestep greater than zero")

    def save(self, file: IO, header="", delimiter=","):
        np.savetxt(
            file,
            np.vstack((self.times, self.dependent_variable.T)).T,
            delimiter=delimiter,
            header=header,
            comments=""
        )

    @property
    def num_dims(self) -> int:
        return self.dependent_variable.shape[1]

    @classmethod
    def from_csv(cls, fp: Union[IO, str, Path], time_index: int = 0):
        data = np.loadtxt(fp, delimiter=",")
        return cls(
            dependent_variable=np.delete(data, obj=time_index, axis=1),
            times=data[:, time_index]
        )

    @property
    def timestep(self) -> float:
        return self.times[1] - self.times[0]

    def __eq__(self, other) -> bool:
        return bool(np.all(self.times == other.times) and np.all(
            np.isclose(self.dependent_variable, other.dependent_variable)
        ))

    def __getitem__(self, key):
        """Enables slicing like time_series[:n]"""
        return TimeSeries(self.dependent_variable[key], self.times[key])

    def __setitem__(self, key, value):
        """Allows modifying slices: time_series[:n] = new_time_series"""
        if not isinstance(value, TimeSeries):
            raise TypeError("Value must be a TimeSeries object")

        if isinstance(key, slice):
            if key.stop is not None and key.stop > len(self.dependent_variable):
                raise ValueError("Slice stop index out of range")
        elif isinstance(key, int):
            if key >= len(self.dependent_variable):
                raise ValueError("Index out of range")

        self.dependent_variable[key] = value.dependent_variable
        self.times[key] = value.times

    def __delitem__(self, key):
        """Allows deleting slices: del time_series[:n]"""
        if isinstance(key, slice):
            if key.stop is not None and key.stop > len(self.dependent_variable):
                raise ValueError("Slice stop index out of range")

            mask = np.ones(len(self.dependent_variable), dtype=bool)
            mask[key] = False
            self.dependent_variable = self.dependent_variable[mask]
            self.times = self.times[mask]

        elif isinstance(key, int):
            if key >= len(self.dependent_variable):
                raise ValueError("Index out of range")

            self.dependent_variable = np.delete(self.dependent_variable, key, axis=0)
            self.times = np.delete(self.times, key, axis=0)
