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
        assert np.isclose(np.std(timesteps), 0.0)

    def save(self, file: IO, header="", delimiter=","):
        np.savetxt(file,
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
