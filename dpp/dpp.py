import numpy as np
import sys

def esym_poly(k, lam):
  N = lam.size
  E = np.zeros((k+1, N+1))
  E[0, :] = np.ones((1, N+1))
  for l in range(1, k+1):
    for n in range(1, N+1):
      E[l, n] = E[l, n-1] + lam[n-1]*E[l-1, n-1]

  return E

def sample_k(k, lam, V_full):
  E = esym_poly(k, lam)
  J = []
  remaining = k-1
  i = lam.size-1

  while remaining>=0:
    marg = 0.0
    if i == remaining:
      marg = 1.0
    else:
      marg = lam[i]*E[remaining, i]/E[remaining+1, i+1]

    if np.random.rand() < marg:
      J.append(i)
      remaining = remaining-1
    
    i = i-1

  k = len(J)-1
  Y = np.zeros((len(J), 1))
  V = V_full[:, J]

  for i in range(k, 0, -1):
    # Sample
    #Pr = 1.0/(float(V.shape[1]))*np.sum(np.power(V, 2), 1)
    #C = np.cumsum(Pr)
    #Y[i] = np.sum((np.random.rand()>C).astype(int))
    Pr = np.sum(V**2, 1)
    Pr = Pr/sum(Pr)
    C = np.cumsum(Pr)
    Y[i] = np.argwhere(np.random.rand() <= C)[0]

    # Update V 
    j = np.argwhere(V[int(Y[i]), :])[0]
    Vj = V[:, j]
    V = np.delete(V, j, 1)
    V = V - np.outer(Vj, V[int(Y[i]), :]/Vj[int(Y[i])])

    # GS orthogonalization
    for a in range(i):
      for b in range(a):
        V[:, a] = V[:, a] - (V[:, a].dot(V[:, b]))*(V[:, b])

      V[:, a] = V[:, a]/np.linalg.norm(V[:, a])

  return Y
