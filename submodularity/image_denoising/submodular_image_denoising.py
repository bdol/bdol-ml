from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import sys

lena = misc.lena().astype(float)/255.0
noisy_mask = np.random.poisson(0.05, lena.shape) > 0
noisy_lena = np.array(lena)
noisy_lena[noisy_mask] = 0
# print lena
# print noisy_lena
# print np.random.poisson(0.1, lena.shape).astype(float)

fig = plt.figure()
plt.gray()
plt.subplot(121)
plt.imshow(lena)
plt.subplot(122)
plt.imshow(noisy_lena)
plt.show()