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

address_matrix = str(sys.argv[1])
spectra_csv = str(sys.argv[2])
threshold = float(sys.argv[3])

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

if classify_user(address_matrix, spectra_csv, threshold, similarity, dim_red):
    print(address_matrix, 'is a member of the class')
else:
    print(address_matrix, 'is not a member of the class')