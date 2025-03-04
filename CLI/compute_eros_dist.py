import numpy as np
import numpy.linalg as npla
import os
import pandas as pd
from dim_red_methods import diffusion_map, pca
import statistics as stat
from similarity_functions import *
from analysis import *
from extract_sender_receiver import *

import sys

# functionality:
# X read in a csv of transaction data into matrices
# X take a single address and csv of transaction data and read into a matrix
# X compute weights (specify dimension reduction method)
# X compute eros distance between two matrices (given weights, similarity function, and dim reduction method)
# X classify address based on threshold value and spectra
# - define spectra for a class given folder of matrices

matrix_1 = str(sys.argv[1])
matrix_2 = str(sys.argv[2])

A = np.genfromtxt(matrix_1, delimiter=',', dtype=np.float64, skip_header=1)
A = np.array([[int(x.decode()) if isinstance(x, bytes) else x for x in row] for row in A]).T 

B = np.genfromtxt(matrix_2, delimiter=',', dtype=np.float64, skip_header=1)
B = np.array([[int(x.decode()) if isinstance(x, bytes) else x for x in row] for row in B]).T 
              
weights_file = str(sys.argv[3])
weights = np.genfromtxt(weights_file, delimiter=',', dtype=np.float64, skip_header=1)
weights = np.array([[int(x.decode()) if isinstance(x, bytes) else x for x in row] for row in weights]).T 

if sys.argv[4]:
    if str(sys.argv[4]) == 'cosine':
        similarity = cosine_similarity
    elif str(sys.argv[4]) == 'euclidean':
        similarity = euclidean_distance
    elif str(sys.argv[4]) == 'mse':
        similarity = mean_squared_error
    elif str(sys.argv[4]) == 'relative_difference':
        similarity = relative_difference
else:
    similarity = cosine_similarity

if sys.argv[5]:
    if str(sys.argv[5]) == 'diffusion_map':
        dim_red = diffusion_map
    elif str(sys.argv[5]) == 'pca':
        dim_red = pca
else:
    dim_red = pca

dist = Eros(A,B,weights,similarity,dim_red)
print('Eros distance:', dist)