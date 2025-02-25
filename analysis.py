import numpy as np
import numpy.linalg as npla
import os
import pandas as pd
from diffusion_map import diffusion_map
import statistics as stat

np.set_printoptions(precision=4, linewidth=200)

def compute_weight_raw(eig_mat):
    n = eig_mat.shape[0]

    weight_raw = np.max(eig_mat, axis=1)
    weight_raw = weight_raw / np.sum(weight_raw)
    
    return np.real(weight_raw)

def compute_weight_norm(eig_mat):
    n = eig_mat.shape[0]

    eig_mat = eig_mat / np.sum(eig_mat, axis=0, keepdims=True)

    return compute_weight_raw(eig_mat)

def cosine_similarity(a, b):
    return np.dot(a,b) / (npla.norm(a) * npla.norm(b))

def Eros_PCA(A, B, weights):
    n = np.shape(A)[0]

    cov_A = np.cov(A)
    cov_B = np.cov(B)

    U_A, S_A, V_A = npla.svd(cov_A)
    U_B, S_B, V_B = npla.svd(cov_B)

    result = 0
    for i in range(n):
        result += weights[i] * np.abs(cosine_similarity(V_A[i], V_B[i]))
    
    return result

def Eros_Diffusion(A, B, weights):
    n = np.shape(A)[0]

    val_A, vec_A = diffusion_map(A)
    val_B, vec_B = diffusion_map(B)

    result = 0

    for i in range(n):
        result += weights[i] * np.abs(cosine_similarity(vec_A[i], vec_B[i]))

    result /= n
    
    return result

def build_eig_mat_diff(folder):
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

    eigs = np.stack(eigenvalues, axis=1)

    return eigs

def build_eig_mat_pca(folder):
    # Iterate over files in directory
    sig_vals = []
    matrices = []
    for name in os.listdir(folder):
        with open(os.path.join(folder, name)) as f:
            A = np.genfromtxt(f, delimiter=',', dtype=np.float64, skip_header=1)
            A = np.array([[int(x.decode()) if isinstance(x, bytes) else x for x in row] for row in A]).T 
            matrices.append(A)
    
    for A in matrices:
        cov_A = np.cov(A.T)
        U_A, S_A, V_A = npla.svd(cov_A)
        sig_vals.append(S_A)


    eigs = np.stack(sig_vals, axis=1)

    return eigs

def compare_distances(folder1, folder2):
    # Construct all the matrices
    mat_1 = []
    mat_2 = []

    for file in os.listdir(folder1):
        with open(os.path.join(folder2, file)) as f:
            A = np.genfromtxt(f, delimiter=',', dtype=np.float64, skip_header=1)
            A = np.array([[int(x.decode()) if isinstance(x, bytes) else x for x in row] for row in A]).T 
            mat_1.append(A)

    for file in os.listdir(folder2):
        with open(os.path.join(folder2, file)) as f:
            A = np.genfromtxt(f, delimiter=',', dtype=np.float64, skip_header=1)
            A = np.array([[int(x.decode()) if isinstance(x, bytes) else x for x in row] for row in A]).T 
            mat_2.append(A)

    # Construct weights for interest class
    eigs_mat_diff = build_eig_mat_diff(folder1)
    eigs_mat_pca = build_eig_mat_pca(folder1)

    weights_diff = compute_weight_norm(eig_mat_diff)
    weights_pca = compute_weight_norm(eig_mat_pca)

    difference_in_same_diff = []
    difference_in_same_pca = []
    for A_1 in mat_1:
        for A_2 in mat_1:
            if not np.all(A_1 == A_2):
                difference_in_same_diff.append(Eros_Diffusion(A_1, A_2, weights_diff))
                difference_in_same_pca.append(Eros_PCA(A_1,A_2, weights_pca))
    
    difference_in_diff_diff = []
    difference_in_diff_pca = []

    for A in mat_1:
        for B in mat_2:
            difference_in_diff_diff.append(Eros_Diffusion(A,B,weights_diff))
            difference_in_diff_pca.append(Eros_PCA(A,B,weights_pca))

    print('--- Diffusion Map ---')
    print('Mean difference in same class:', stat.mean(difference_in_same_diff), ', Standard deviation:', stat.stdev(difference_in_same_diff))
    print('Mean difference in different class:', stat.mean(difference_in_diff_diff), ', Standard deviation:', stat.stdev(difference_in_diff_diff))
    print('-------- PCA --------')
    print('Mean difference in same class:', stat.mean(difference_in_same_pca), ', Standard deviation:', stat.stdev(difference_in_same_pca))
    print('Mean difference in different class:', stat.mean(difference_in_diff_pca), ', Standard deviation:', stat.stdev(difference_in_diff_pca))
    print('---------------------')

f = "Matrices/0x0a05956d2e3a21379af4abaa17bf883c04a67a7e.csv"
A = np.genfromtxt(f, delimiter=',', dtype=np.float64, skip_header=1)
A = np.array([[int(x.decode()) if isinstance(x, bytes) else x for x in row] for row in A]).T 

f = "Matrices/0xfdc27cb5c94095b2877ed9f688dd7a39d2bf45cd.csv"
B = np.genfromtxt(f, delimiter=',', dtype=np.float64, skip_header=1)
B = np.array([[int(x.decode()) if isinstance(x, bytes) else x for x in row] for row in B]).T 

f = "Matrices/0xa611438e5637c227e5080477b7180fc3d1c76710f2aa5f6d71c28a1dee1e2ed4.csv"
C = np.genfromtxt(f, delimiter=',', dtype=np.float64, skip_header=1)
C = np.array([[int(x.decode()) if isinstance(x, bytes) else x for x in row] for row in C]).T 

eig_mat_diff = build_eig_mat_diff('Matrices')
weights_diff = compute_weight_norm(eig_mat_diff)

print(Eros_Diffusion(A, B, weights_diff))
print(Eros_Diffusion(A, C, weights_diff))

eig_mat_pca = build_eig_mat_diff('Matrices')
weights_pca = compute_weight_norm(eig_mat_pca)