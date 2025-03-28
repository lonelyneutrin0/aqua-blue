import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import aqua_blue


def lotka_volterra(t, z, alpha, beta, delta, gamma):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]


def solve_lv(t_start, t_end, no, alpha=0.1, beta=0.02, gamma=0.3, delta=0.01, x0=20, y0=9):
    
    t_eval = np.linspace(t_start, t_end, no)
    solution = solve_ivp(lotka_volterra, [t_start, t_end], [x0, y0], t_eval=t_eval, args=(alpha, beta, delta, gamma))
    x, y = solution.y
    lotka_volterra_array = np.vstack((x, y)).T
    return lotka_volterra_array


def main():

    y = solve_lv(0, 10, 1000)
    t = np.linspace(0, 10, 1000)
    
    time_series = aqua_blue.time_series.TimeSeries(dependent_variable=y, times=t)
    
    normalizer = aqua_blue.utilities.Normalizer()
    time_series = normalizer.normalize(time_series)
    
    model = aqua_blue.models.Model(
        reservoir=aqua_blue.reservoirs.DynamicalReservoir(
            reservoir_dimensionality=100,
            input_dimensionality=2
        ),
        readout=aqua_blue.readouts.LinearReadout()
    )
    model.train(time_series)

    prediction = model.predict(horizon=1_000)
    prediction = normalizer.denormalize(prediction)
    
    actual_future = solve_lv(prediction.times[0], prediction.times[-1], 1_000)
    
    plt.plot(prediction.times, actual_future)
    plt.xlabel('t')
    plt.plot(prediction.times, prediction.dependent_variable)
    plt.legend(['actual x', 'actual y', 'predicted x', 'predicted y'], shadow=True)
    plt.title('Lotka-Volterra System')
    plt.show()


if __name__ == "__main__":

    main()

 
