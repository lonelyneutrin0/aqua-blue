from dataclasses import dataclass, field

from numpy.typing import NDArray

from .time_series import TimeSeries


@dataclass
class Normalizer:

    means: NDArray = field(init=False)
    standard_deviations: NDArray = field(init=False)

    def normalize(self, time_series: TimeSeries) -> TimeSeries:

        if hasattr(self, "means") or hasattr(self, "standard_deviations"):
            raise ValueError("You can only use the Normalizer once. Create a new instance to normalize again")

        arr = time_series.dependent_variable
        self.means = arr.mean(axis=0)
        self.standard_deviations = arr.std(axis=0)

        arr = arr - self.means
        arr = arr / self.standard_deviations

        return TimeSeries(
            dependent_variable=arr,
            times=time_series.times
        )

    def denormalize(self, time_series: TimeSeries) -> TimeSeries:

        if not hasattr(self, "means") or not hasattr(self, "standard_deviations"):
            raise ValueError("You can only denormalize after normalizing a time series")

        arr = time_series.dependent_variable
        arr = arr * self.standard_deviations
        arr = arr + self.means

        return TimeSeries(dependent_variable=arr, times=time_series.times)
