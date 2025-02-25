import numpy as np
import numpy.linalg as npla

### SIMILARITY FUNCTIONS ###
def cosine_similarity(a, b):
    return np.dot(a,b) / (npla.norm(a) * npla.norm(b))

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def mean_squared_error(a, b):
    return np.mean((a - b) ** 2)

def relative_difference(a, b):
    return np.mean(np.abs((a - b) / a)) * 100
############################