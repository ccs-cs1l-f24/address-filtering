import numpy as np
import numpy.linalg as npla
import os
import pandas as pd
from dim_red_methods import diffusion_map, pca
import statistics as stat
from similarity_functions import *

np.set_printoptions(precision=4, linewidth=200)

#### CALCULATE WEIGHTS ####
def compute_weight_raw(eig_mat):
    n = eig_mat.shape[0]

    weight_raw = np.max(eig_mat, axis=1)
    weight_raw = weight_raw / np.sum(weight_raw)
    
    return np.real(weight_raw)

def compute_weight_norm(eig_mat):
    n = eig_mat.shape[0]

    eig_mat = eig_mat / np.sum(eig_mat, axis=0, keepdims=True)

    return compute_weight_raw(eig_mat)
############################

#### EROS DISTANCE ######
def Eros(A,B,weights,similarity,dim_red):
    n = np.shape(A)[0]

    val_A, vec_A = dim_red(A)
    val_B, vec_B = dim_red(B)

    result = 0

    for i in range(n):
        result += weights[i] * np.abs(similarity(vec_A[i], vec_B[i]))

    result /= n
    
    return result
############################

#### MATRIX OF EIGENVALUES ######
def build_eig_mat(matrices, dim_red):
    # Iterate over files in directory
    eigenvalues = []
    eigenvectors = []
    
    for A in matrices:
        val, vec = dim_red(A)
        eigenvalues.append(val)
        eigenvectors.append(vec.T)

    eigs = np.stack(eigenvalues, axis=1)
    vecs = [np.stack([vec[i] for vec in eigenvectors], axis=1) for i in range(eigenvectors[0].shape[0])]

    return eigs, vecs
##################################

def define_spectra(folder1, dim_red, func=np.mean):
    # construct the matrices
    mat_1 = []
    for file in os.listdir(folder1):
        with open(os.path.join(folder1, file)) as f:
            A = np.genfromtxt(f, delimiter=',', dtype=np.float64, skip_header=1)
            A = np.array([[int(x.decode()) if isinstance(x, bytes) else x for x in row] for row in A]).T 
            mat_1.append(A)

    # construct weights for class of interest
    eigs_mat, eigs_vec_list = build_eig_mat(mat_1, dim_red)

    centroids = np.stack([func(V, axis=1) for V in eigs_vec_list], axis=1)
    return centroids


def compare_distances(folder1, folder2, similarity, dim_red_1, dim_red_2, max_iter=100):
    # construct the matrices
    mat_1 = []
    mat_2 = []

    num = 0
    for file in os.listdir(folder1):
        if num < max_iter:
            with open(os.path.join(folder1, file)) as f:
                A = np.genfromtxt(f, delimiter=',', dtype=np.float64, skip_header=1)
                A = np.array([[int(x.decode()) if isinstance(x, bytes) else x for x in row] for row in A]).T 
                mat_1.append(A)
                num += 1
        else:
            break
    print('100 class matrices constructed...')

    num = 0
    for file in os.listdir(folder2):
        if num < max_iter:
            with open(os.path.join(folder2, file)) as f:
                A = np.genfromtxt(f, delimiter=',', dtype=np.float64, skip_header=1)
                A = np.array([[int(x.decode()) if isinstance(x, bytes) else x for x in row] for row in A]).T 
                mat_2.append(A)
                num += 1
        else:
            break
    print('100 comparison matrices constructed...')

    # construct weights for class of interest
    eigs_mat_diff, vec = build_eig_mat(mat_1, dim_red_1)
    eigs_mat_pca, vec = build_eig_mat(mat_1, dim_red_2)
    print('Eigenvalues calculated...')

    weights_diff = compute_weight_norm(eigs_mat_diff)
    weights_pca = compute_weight_norm(eigs_mat_pca)
    print('Weights found...')

    difference_in_same_diff = []
    difference_in_same_pca = []
    for A_1 in mat_1:
        for A_2 in mat_1:
            if not np.all(A_1 == A_2):
                difference_in_same_diff.append(Eros(A_1, A_2, weights_diff, similarity, dim_red_1))
                difference_in_same_pca.append(Eros(A_1,A_2, weights_pca, similarity, dim_red_2))
    print('Distances compared for same class...')

    difference_in_diff_diff = []
    difference_in_diff_pca = []

    for A in mat_1:
        for B in mat_2:
            difference_in_diff_diff.append(Eros(A,B,weights_diff, similarity, dim_red_1))
            difference_in_diff_pca.append(Eros(A,B,weights_pca, similarity, dim_red_1))

    print('--- Diffusion Map ---')
    print('Mean difference in same class:', stat.mean(difference_in_same_diff), ', Standard deviation:', stat.stdev(difference_in_same_diff))
    print('Mean difference in different class:', stat.mean(difference_in_diff_diff), ', Standard deviation:', stat.stdev(difference_in_diff_diff))
    print('-------- PCA --------')
    print('Mean difference in same class:', stat.mean(difference_in_same_pca), ', Standard deviation:', stat.stdev(difference_in_same_pca))
    print('Mean difference in different class:', stat.mean(difference_in_diff_pca), ', Standard deviation:', stat.stdev(difference_in_diff_pca))
    print('---------------------')

# compare_distances('Matrices', 'CompareMatrices', euclidean_distance, diffusion_map, pca)

def classify_user(address_matrix, spectra_csv, threshold, weights, similarity, dim_red):
    A = np.genfromtxt(address_matrix, delimiter=',', dtype=np.float64, skip_header=1)
    A = np.array([[int(x.decode()) if isinstance(x, bytes) else x for x in row] for row in A]).T 
    
    spectra = np.genfromtxt(spectra_csv, delimiter=',', dtype=np.float64, skip_header=1)
    spectra = np.array([[int(x.decode()) if isinstance(x, bytes) else x for x in row] for row in spectra]).T 
    
    dist = Eros(A, spectra, weights, similarity, dim_red)
    
    if dist < threshold:
        return True
    else:
        return False