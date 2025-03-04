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
# - compute eros distance between two matrices (given weights, similarity function, and dim reduction method)
# X classify address based on threshold value and spectra
# - define spectra for a class given folder of matrices

matrices_folder = str(sys.argv[1])
norm_or_not = str(sys.argv[2]) == 'norm'

if sys.argv[3]:
    if str(sys.argv[3]) == 'diffusion_map':
        dim_red = diffusion_map
    elif str(sys.argv[3]) == 'pca':
        dim_red = pca
else:
    dim_red = pca

output_file = str(sys.argv[4])

eig_mat, eig_vec_mat = build_eig_mat(matrices_folder, dim_red)
if norm_or_not:
    weights = compute_weight_norm(eig_mat)
else:
    weights = compute_weight_raw(eig_mat)

np.savetxt(output_file + '.csv', weights, delimiter=",")

print('Weights have been calculated and saved to', output_file + '.csv')
print(weights)