import numpy as np
from dim_red_methods import *

A = np.array([[1,2,3],[4,5,6]])
B = 2 * A + 5

a_val, a_vec = diffusion_map(A)
b_val, b_vec = diffusion_map(B)
print(a_vec)
print(b_vec)

print(np.stack((a_vec,b_vec), axis=0))