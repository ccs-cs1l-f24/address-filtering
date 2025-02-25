import numpy as np
import numpy.linalg as npla
import os
import pandas as pd

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
    # pairwise kernel matrix
    K = kernel_matrix(X, sigma)

    # degree matrix
    D = np.diag(1/np.sqrt(np.sum(K, axis=1)))

    # normalized diffusion matrix (diffusion process)
    P = np.dot(np.dot(D, K), D)

    # get "principal components" in a way
    w, v = npla.eig(P)

    # sort
    idx = np.argsort(w)[::-1]

    # eigenvalues
    w = w[idx]
    # eigenvectors
    v = v[:, idx]

    return w, v