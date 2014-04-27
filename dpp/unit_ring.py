import matplotlib.pyplot as plt
import numpy as np
import dpp
import math

t = 0.005
x = []
y = []
for i in range(0, int(1/t)):
  a = 2*math.pi*float(i)*t
  x.append(math.cos(a))
  y.append(math.sin(a))

x = np.array(x)
y = np.array(y)

k = 50
idx = np.arange(x.size)
np.random.shuffle(idx)
x_uniform = x[idx[:k]]
y_uniform = y[idx[:k]]

sigma = 0.1
L = np.exp(- ( np.power(x - x[:, None], 2) + 
               np.power(y - y[:, None], 2) )/(sigma**2))
D, V = np.linalg.eig(L.T)
Y = dpp.sample_k(k, np.real(D), np.real(V))

L = np.power(np.outer(x, x)+np.outer(y, y), 2)
D, V = np.linalg.eig(L.T)
Y2 = dpp.sample_k(k, np.real(D), np.real(V))

plt.figure(1)
plt.subplot(1, 3, 1)
plt.plot(x_uniform, y_uniform, 'ro')
plt.xlim([-1.2, 1.2])
plt.ylim([-1.2, 1.2])
plt.title('Uniform')

plt.subplot(1, 3, 2)
plt.title('DPP (Gaussian Similarity)')
plt.plot(x[Y.astype(int)], y[Y.astype(int)], 'bo')
plt.xlim([-1.2, 1.2])
plt.ylim([-1.2, 1.2])

plt.subplot(1, 3, 3)
plt.title('DPP (Squared Cosine Similarity)')
plt.plot(x[Y2.astype(int)], y[Y2.astype(int)], 'bo')
plt.xlim([-1.2, 1.2])
plt.ylim([-1.2, 1.2])

plt.show()
