import numpy as np
import numpy.linalg as npla
import os
import pandas as pd
from diffusion_map import diffusion_map

np.set_printoptions(precision=4, linewidth=200)

def cosine_similarity(a, b):
    return np.dot(a,b) / (npla.norm(a) * npla.norm(b))

def Eros(A, B, weights):
    n = np.shape(A)[0]

    val_A, vec_A = diffusion_map(A)
    val_B, vec_B = diffusion_map(B)

    result = 0
    print(vec_A.shape)
    for i in range(n):
        result += weights[i] * np.abs(cosine_similarity(vec_A[i], vec_B[i]))

    result /= n
    
    return result

def define_weights(folder):
    # Iterate over files in directory
    eigenvalues = []
    matrices = []
    for name in os.listdir(folder):
        with open(os.path.join(folder, name)) as f:
            A = np.genfromtxt(f, delimiter=',', dtype=np.float64, skip_header=1)
            A = np.array([[int(x.decode()) if isinstance(x, bytes) else x for x in row] for row in A]).T 
            matrices.append(A)
    
    for A in matrices:
        val, vec = diffusion_map(A)
        eigenvalues.append(val)
        print(val)

    eigs = np.stack(eigenvalues, axis=0)
    means = np.real(np.mean(eigs, axis=0))

    return means
f = "Matrices/0x0a05956d2e3a21379af4abaa17bf883c04a67a7e.csv"
A = np.genfromtxt(f, delimiter=',', dtype=np.float64, skip_header=1)
A = np.array([[int(x.decode()) if isinstance(x, bytes) else x for x in row] for row in A]).T 

f = "Matrices/0xfdc27cb5c94095b2877ed9f688dd7a39d2bf45cd.csv"
B = np.genfromtxt(f, delimiter=',', dtype=np.float64, skip_header=1)
B = np.array([[int(x.decode()) if isinstance(x, bytes) else x for x in row] for row in B]).T 

f = "transaction_data.csv"
C = np.genfromtxt(f, delimiter=',', dtype=np.float64, skip_header=1)
C = np.array([[int(x.decode()) if isinstance(x, bytes) else x for x in row] for row in C]).T 

weights = define_weights('Matrices')

print(Eros(A, B, weights))
print(Eros(A, C, weights))

# to do: modify weight calculation using description in Eros Algorithm paper