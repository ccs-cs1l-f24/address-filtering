import numpy as np
import numpy.linalg as npla
import os
import pandas as pd

# f = 'Code-Files/time_series_data/series_0xd433138d12beB9929FF6fd583DC83663eea6Aaa5.csv'
# A = np.genfromtxt(f, delimiter=',', dtype=np.float64, skip_header=1)
# A = np.array([[int(x.decode()) if isinstance(x, bytes) else x for x in row] for row in A]) 

# f = 'Code-Files/time_series_data/series_0x9B99CcA871Be05119B2012fd4474731dd653FEBe.csv'
# B = np.genfromtxt(f, delimiter=',', dtype=np.float64, skip_header=1)
# B = np.array([[int(x.decode()) if isinstance(x, bytes) else x for x in row] for row in B]) 


# f = 'Code-Files/time_series_data/series_0x4838B106FCe9647Bdf1E7877BF73cE8B0BAD5f97.csv'
# C = np.genfromtxt(f, delimiter=',', dtype=np.float64, skip_header=1)
# C = np.array([[int(x.decode()) if isinstance(x, bytes) else x for x in row] for row in C])

def kernel_func(x, y, sigma):
    return np.exp(-npla.norm(x-y)**2/(2*sigma**2))

def kernel_matrix(X, sigma):
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel_func(X[i], X[j], sigma)
    return K

def diffusion_map(X, sigma):
    K = kernel_matrix(X, sigma)
    D = np.diag(1/np.sqrt(np.sum(K, axis=1)))
    P = np.dot(np.dot(D, K), D)
    w, v = npla.eig(P)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:, idx]
    return w, v