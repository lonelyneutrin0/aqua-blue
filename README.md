# aqua-blue
Lightweight and basic reservoir computing library

[![PyPI version shields.io](https://img.shields.io/pypi/v/aqua-blue.svg)](https://pypi.python.org/pypi/aqua-blue/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/aqua-blue.svg)](https://pypi.python.org/pypi/aqua-blue/)

## 🌊 What is aqua-blue?

`aqua-blue` is a lightweight `python` library for reservoir computing (specifically [echo state networks](https://en.wikipedia.org/wiki/Echo_state_network)) depending only on `numpy`. `aqua-blue`'s namesake comes from:

- A blue ocean of data, aka a reservoir 💧
- A very fancy cat named Blue 🐾

## 📥 Installation

`aqua-blue` is on PyPI, and can therefore be installed with `pip`:

```bash
pip install aqua-blue
```

## 📝 Quickstart

```py
import numpy as np
from aqua_blue import TimeSeries, EchoStateNetwork

# generate arbitrary two-dimensional time series
# y_1(t) = cos(t), y_2(t) = sin(t)
# resulting dependent variable has shape (number of timesteps, 2)
t = np.linspace(0, 4.0 * np.pi, 10_000)
y = np.vstack((np.cos(t), np.sin(t))).T

# create time series object to feed into echo state network
time_series = TimeSeries(dependent_variable=y, times=t)

# generate echo state network with a relatively high reservoir dimensionality
esn = EchoStateNetwork(reservoir_dimensionality=100, input_dimensionality=2)

# train esn on our time series
esn.train(time_series)

# predict 1,000 steps into the future
prediction = esn.predict(horizon=1_000)
```

## 📃 License

`aqua-blue` is released under the MIT License.

---

![Blue](https://raw.githubusercontent.com/Chicago-Club-Management-Company/aqua-blue/refs/heads/main/assets/blue.jpg)

*Blue, the cat behind `aqua-blue`.*