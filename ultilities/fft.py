from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
N = 600
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N, endpoint=False)
dpi = 2.0*np.pi
y = np.sin(50.0 * dpi*x) + 0.5*np.sin(80.0 *dpi*x)
yf = fft(y)
xf = fftfreq(N, T)[:N//2]
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()