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
# - read in a csv of transaction data into matrices
# - take a single address and csv of transaction data and read into a matrix
# - compute weights (specify dimension reduction method)
# - compute eros distance between two matrices (given weights, similarity function, and dim reduction method)
# - classify address based on threshold value and spectra
# - define spectra for a class given folder of matrices

matrix_folder_path = str(sys.argv[1])
if sys.argv[2]:
    if str(sys.argv[2]) == 'diffusion_map':
        dim_red = diffusion_map
    elif str(sys.argv[2]) == 'pca':
        dim_red = pca
else:
    dim_red = pca

if sys.argv[3]:
    if str(sys.argv[3]) == 'max':
        func = np.max
    elif str(sys.argv[3]) == 'min':
        func = np.min
    elif str(sys.argv[3]) == 'mean':
        func = np.mean
else:
    func = np.mean

try:
    output_folder = float(sys.argv[4])
except:
    output_folder = 'UserSpectra.csv'

centroids = define_spectra(matrix_folder_path, dim_red, func)
np.savetxt(output_folder, centroids, delimiter=",")

print('Spectra for the class have been constructed and saved to', output_folder)