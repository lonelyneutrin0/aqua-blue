import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

from aqua_blue.time_series import TimeSeries 
from aqua_blue.utilities import Normalizer 
from aqua_blue.models import Model 
from aqua_blue.reservoirs import DynamicalReservoir 
from aqua_blue.readouts import LinearReadout 

@dataclass 
class SpaceParameters: 
    L: float = 64 
    N: int = 128

@dataclass
class TimeParameters: 
    dt: float = 0.05 
    tmax: float = 100.0


def kuramoto_sivashinsky(space_: SpaceParameters, time_: TimeParameters) -> np.typing.NDArray:
    """ Approximate solution to the Kuramoto-Sivashinsky equation using ETDRK4 """
    N = space_.N              
    L = space_.L         
    
    x = np.linspace(0, L, N, endpoint=False)
    a = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    k = 1j * a
    
    dt = time_.dt           
    T = time_.tmax        
    nsteps = int(T / dt)
    
    u = 0.1 * np.random.randn(N)
    v = np.fft.fft(u)
    
    Lhat = -k**2 - k**4
    E = np.exp(Lhat * dt)
    E2 = np.exp(Lhat * dt / 2)
    
    M = 32
    r = np.exp(1j * np.pi * (np.arange(1, M+1) - 0.5) / M)
    LR = dt * Lhat[:, None] + r
    Q = dt * np.mean((np.exp(LR / 2) - 1) / LR, axis=1)
    f1 = dt * np.mean((-4 - LR + np.exp(LR)*(4 - 3*LR + LR**2)) / LR**3, axis=1)
    f2 = dt * np.mean((2 + LR + np.exp(LR)*(-2 + LR)) / LR**3, axis=1)
    f3 = dt * np.mean((-4 - 3*LR - LR**2 + np.exp(LR)*(4 - LR)) / LR**3, axis=1)
    
    t_output = np.linspace(0, T, 400)
    usave = []
    j = 0
    
    # Dealiasing mask (2/3 rule)
    kcut = N // 3
    dealias = np.zeros(N)
    dealias[:kcut] = 1
    dealias[-kcut:] = 1

    # Time stepping
    for i in range(nsteps):
        u_phys = np.fft.ifft(v).real
        Nv = -0.5j * k * np.fft.fft(u_phys ** 2)
        Nv *= dealias

        a = E2 * v + Q * Nv
        Na = -0.5j * k * np.fft.fft(np.fft.ifft(a).real ** 2)
        Na *= dealias

        b = E2 * v + Q * Na
        Nb = -0.5j * k * np.fft.fft(np.fft.ifft(b).real ** 2)
        Nb *= dealias

        c = E2 * a + Q * (2*Nb - Nv)
        Nc = -0.5j * k * np.fft.fft(np.fft.ifft(c).real ** 2)
        Nc *= dealias

        v = E * v + Nv * f1 + 2*(Na + Nb) * f2 + Nc * f3

        # if i == output_idx[j]:
        usave.append(np.fft.ifft(v).real.copy())
            # j += 1
    
    return np.array(usave)

def main():
    # Model Params 
    LEAKING_RATE = 0.35
    SPECTRAL_RADIUS = 0.95
    RIDGE_COEFFICIENT = 1e-4

    # Space Params
    L = 64
    N = 128
    space_ = SpaceParameters(L=L, N=N)

    # Time Params
    DT = 0.05
    T_MAX = 100
    time_ = TimeParameters(dt=DT, tmax=T_MAX)

    HORIZON = 250

    snapshots = kuramoto_sivashinsky(space_, time_)
    print(snapshots.shape)

    TRAIN_DATA = snapshots[:snapshots.shape[0]-(HORIZON-1), :]
    TEST_DATA = snapshots[-HORIZON:, :]

    t = np.arange(0, T_MAX, DT)
    ts = TimeSeries(
        dependent_variable=TRAIN_DATA, 
        times=t
    )

    normalizer = Normalizer()
    time_series = normalizer.normalize(ts)

    model = Model(
        reservoir=DynamicalReservoir(
            input_dimensionality=128,
            reservoir_dimensionality=100,
            leaking_rate=LEAKING_RATE,
            spectral_radius=SPECTRAL_RADIUS,
        ),
        readout= LinearReadout(rcond=RIDGE_COEFFICIENT)
    )

    model.train(time_series)
    predictions = model.predict(HORIZON)
    predictions = normalizer.denormalize(predictions)

    print(f'RMSE: {np.mean(np.sqrt((TEST_DATA - predictions.dependent_variable)**2), axis=(0, 1))}')

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axs[0].imshow(snapshots.T, aspect='auto', extent=[0, time_.tmax, 0, space_.L], cmap='inferno', origin='lower')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Space')
    axs[0].set_title('Kuramoto-Sivashinsky Equation: Spatiotemporal Chaos [Actual]')
    axs[0].axvline(x=T_MAX*(1 - HORIZON/snapshots.shape[0]), color='black', linestyle='--', linewidth=2, label='Prediction Horizon')
    fig.colorbar(im0, ax=axs[0], label='u(x,t)')

    im1 = axs[1].imshow(np.vstack((TRAIN_DATA, predictions.dependent_variable)).T, aspect='auto', extent=[0, time_.tmax, 0, space_.L], cmap='inferno', origin='lower')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Space')
    axs[1].set_title('Kuramoto-Sivashinsky Equation: Spatiotemporal Chaos [Predicted]')
    axs[1].axvline(x=T_MAX*(1 - HORIZON/snapshots.shape[0]), color='black', linestyle='--', linewidth=2)
    fig.colorbar(im1, ax=axs[1], label='u(x,t)')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
