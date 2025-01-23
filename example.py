import numpy as np

from aqua_blue import TimeSeries, EchoStateNetwork


def main():

    # seed for reproducibility
    np.random.seed(0)

    # generate arbitrary two-dimensional time series
    # y_1(t) = cos(t), y_2(t) = sin(t)
    # resulting dependent variable has shape (number of timesteps, 2)
    t = np.linspace(0, 4.0 * np.pi, 10_000)
    y1 = np.cos(t)
    y2 = np.sin(t)
    y = np.vstack((y1, y2)).T

    # create time series object to feed into echo state network
    time_series = TimeSeries(
        dependent_variable=y,
        times=t
    )

    # generate echo state network with a relatively high reservoir dimensionality
    esn = EchoStateNetwork(
        reservoir_dimensionality=100,
        input_dimensionality=2
    )

    # train esn on our time series
    esn.train(time_series)

    # predict 1,000 steps into the future
    prediction = esn.predict(horizon=1_000)

    # test prediction since we know the next 10_000 timesteps
    # benchmark with root-mean-square error
    actual_future = np.vstack((
        np.cos(prediction.times), np.sin(prediction.times)
    )).T

    root_mean_square_error = np.sqrt(
        np.mean((actual_future - prediction.dependent_variable) ** 2)
    )
    print(root_mean_square_error)


if __name__ == "__main__":

    main()
