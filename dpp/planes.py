import matplotlib.pyplot as plt
import numpy as np
import dpp
import sys

# Width, height of the sampling grid
n = 50
x, y = np.meshgrid(range(1, n+1), range(1, n+1))
x = 1/float(n)*(x.flatten())
y = 1/float(n)*(y.flatten())

# Number of samples to generate
k = 60

# Randomly sample k points
idx = np.arange(x.size)
np.random.shuffle(idx)
x_uniform = x[idx[:k]]
y_uniform = y[idx[:k]]

# Sample a k-DPP
# First construct a Gaussian L-ensemble
sigma = 0.1
L = np.exp(- ( np.power(x - x[:, None], 2) + 
               np.power(y - y[:, None], 2) )/(sigma**2))
D, V = dpp.decompose_kernel(L)
Y = dpp.sample_k(k, D, V)
print "Done Gaussian!"

L = np.power(np.outer(x, x)+np.outer(y, y), 2)
D, V = dpp.decompose_kernel(L)
Y2 = dpp.sample_k(k, D, V)

# Plot both
plt.figure(1)
plt.subplot(1, 3, 1)
plt.plot(x_uniform, y_uniform, 'ro')
plt.title('Uniform')

plt.subplot(1, 3, 2)
plt.title('DPP (Gaussian Similarity)')
plt.plot(x[Y.astype(int)], y[Y.astype(int)], 'bo')

plt.subplot(1, 3, 3)
plt.title('DPP (Cosine Similarity)')
plt.plot(x[Y2.astype(int)], y[Y2.astype(int)], 'bo')
plt.show()
