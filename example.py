import numpy as np
from aqua_blue import TimeSeries, EchoStateNetwork, utilities


def main():

    # generate arbitrary two-dimensional time series
    # y_1(t) = cos(t), y_2(t) = sin(t)
    # resulting dependent variable has shape (number of timesteps, 2)
    t = np.linspace(0, 4.0 * np.pi, 10_000)
    y = np.vstack((2.0 * np.cos(t) + 1, 5.0 * np.sin(t) - 1)).T

    # create time series object to feed into echo state network
    time_series = TimeSeries(dependent_variable=y, times=t)

    # normalize
    normalizer = utilities.Normalizer()
    time_series = normalizer.normalize(time_series)

    # generate echo state network with a relatively high reservoir dimensionality
    esn = EchoStateNetwork(reservoir_dimensionality=100, input_dimensionality=2)

    # train esn on our time series
    esn.train(time_series)

    # predict 1,000 steps into the future
    prediction = esn.predict(horizon=1_000)

    # denormalize
    prediction = normalizer.denormalize(prediction)

    actual_future = np.vstack((
        (2.0 * np.cos(prediction.times) + 1, 5.0 * np.sin(prediction.times) - 1)
    )).T
    root_mean_square_error = np.sqrt(
        np.mean((actual_future - prediction.dependent_variable) ** 2)
    )
    print(root_mean_square_error)


if __name__ == "__main__":

    main()
