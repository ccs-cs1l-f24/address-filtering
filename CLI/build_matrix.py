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
# - compute weights (specify dimension reduction method)
# - compute eros distance between two matrices (given weights, similarity function, and dim reduction method)
# - classify address based on threshold value and spectra
# - define spectra for a class given folder of matrices

path = str(sys.argv[1])
address = str(sys.argv[2])
try:
    output_folder = float(sys.argv[3])
except:
    output_folder = 'Matrices'

construct_single_matrix(path, address, output_folder)
print('Matrix has been constructed and saved to', output_folder + '/' + address + '.csv')