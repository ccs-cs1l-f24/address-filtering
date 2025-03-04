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
# - take a single address and csv of transaction data and read into a matrix (NEED TO DO)
# - compute weights (specify dimension reduction method)
# - compute eros distance between two matrices (given weights, similarity function, and dim reduction method)
# - classify address based on threshold value (NEED TO DO)
# - define spectra for a class given folder of matrices
