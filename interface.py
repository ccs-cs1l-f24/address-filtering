import os
import argparse
 
from extract_sender_receiver import *
from analysis import *
from dim_red_methods import *
from similarity_functions import *


def main():
	parser = argparse.ArgumentParser(description='Extract and analyze Ethereum transaction data')

	parser.add_argument("-b", "--build", type = str, nargs = 3,
						metavar =("path", "address", "output_to"), default = None,
						help = "Build a matrix from a given address.")
    
	parser.add_argument("-bf", "--build-folder", type = str, nargs = 3,
						metavar = ("path", "blacklist", "output_to"), default = None,
						help = "Build a folder of matrices in list of addresses of interest given a transaction file.")
	
	parser.add_argument("-cf", "--comp-build-folder", type = str, nargs = 3,
						metavar = ("path", "blacklist", "output_to"), default = None,
						help = "Build a folder of matrices not in list of addresses of interest given a transaction file.")
    
	parser.add_argument("-e", "--eig-matrix", type = str, nargs = 2,
						metavar = ("matrix_folder", "dim_red_method"), default = None,
						help = "Build a matrix of eigenvalues given a folder of matrices and a dimensionality reduction method.")
    
	parser.add_argument("-w", "--calc-weights", type = str, nargs = 3,
						metavar = ("eig_mat_file", 'norm_or_not', "output_to"), default = None,
						help = "Calculate the weights based on a given eigenvalue matrix.")
    
	parser.add_argument("-s", "--define-spectra", type = str, nargs = 3,
						metavar = ("folder", "dim_red_method", "agg_func"), default = None,
						help = "Define spectra for a folder of matrices.")
	
	parser.add_argument("-ed", "--eros-dist", type = str, nargs = 5,
						metavar = ("A", "B", "weights", "similarity", "dim_red_method"), default = None,
						help = "Calculate the Eros distance between two matrices.")
	
	parser.add_argument("-c", "--classify", type = str, nargs = 6,
						metavar = ("addr_mat", "spectra_file", "threshold", "weights", "similarity", "dim_red_method"), default = None,
						help = "Classify a user based on matrix (most likely class spectra).")
	
	args = parser.parse_args()

	if args.build:
		construct_single_matrix(args.build[0], args.build[1], args.build[2])
		print("Matrix constructed and outputted to", args.build[2])
	elif args.build_folder:
		construct_matrix(args.build_folder[0], args.build_folder[1], args.build_folder[2])
		print("Folder of matrices constructed and outputted to", args.build_folder[2])
	elif args.comp_build_folder:
		construct_matrix_not_interest(args.comp_build_folder[0], args.comp_build_folder[1], args.comp_build_folder[2])
		print("Folder of matrices constructed and outputted to", args.comp_build_folder[2])
	elif args.eig_matrix:
		matrix_folder = args.eig_matrix[0]
		dim_red_method = args.eig_matrix[1]

		if dim_red_method == 'diffusion':
			dim_red_method = diffusion_map
		elif dim_red_method == 'pca':
			dim_red_method = pca
		else:
			dim_red_method = diffusion_map

		matrices = []
		for file in os.listdir(matrix_folder):
			if file.endswith('.csv'):
				file_path = os.path.join(matrix_folder, file)
				A = np.genfromtxt(file_path, delimiter=',', dtype=np.float64, skip_header=1)
				A = np.array([[int(x.decode()) if isinstance(x, bytes) else x for x in row] for row in A]).T 
				matrices.append(A)
		eigs, vecs = build_eig_mat(matrices, dim_red_method)
		np.save('eig_mat.npy', eigs)
		np.save('eig_vec.npy', vecs)

		print("Eigenvalue matrix constructed and outputted to eig_mat.npy")
		print("Eigenvector matrix constructed and outputted to eig_vec.npy")
	elif args.calc_weights:
		eig_mat = np.load(args.calc_weights[0])
		norm_or_not = args.calc_weights[1].lower() == 'norm'
		output_to = args.calc_weights[2]
		weights = compute_weight_norm(eig_mat) if norm_or_not else compute_weight_raw(eig_mat)
		np.save(output_to, weights)
		print("Weights calculated and outputted to", output_to)
	elif args.define_spectra:
		if args.define_spectra[2] == 'mean':
			agg_func = np.mean
		elif args.define_spectra[2] == 'median':
			agg_func = np.median
		elif args.define_spectra[2] == 'max':
			agg_func = np.max
		elif args.define_spectra[2] == 'min':
			agg_func = np.min
		else:
			agg_func = np.mean

		if args.define_spectra[1] == 'diffusion':
			dim_red_method = diffusion_map
		else:
			dim_red_method = pca
		spectra = define_spectra(args.define_spectra[0], dim_red_method, agg_func)
		np.save('spectra.npy', spectra)
		print("Spectra defined and outputted to spectra.npy")
	elif args.eros_dist:
		A = np.genfromtxt(args.eros_dist[0], delimiter=',', dtype=np.float64, skip_header=1)
		A = np.array([[int(x.decode()) if isinstance(x, bytes) else x for x in row] for row in A]).T 

		B = np.genfromtxt(args.eros_dist[0], delimiter=',', dtype=np.float64, skip_header=1)
		B = np.array([[int(x.decode()) if isinstance(x, bytes) else x for x in row] for row in B]).T 
				
		weights = np.load(args.eros_dist[2])
		similarity = args.eros_dist[3]
		dim_red_method = args.eros_dist[4]

		if similarity == 'euclidean':
			similarity = euclidean_distance
		elif similarity == 'cosine':
			similarity = cosine_similarity
		elif similarity == 'mse':
			similarity = mean_squared_error

		if dim_red_method == 'diffusion':
			dim_red_method = diffusion_map
		elif dim_red_method == 'pca':
			dim_red_method = pca

		distance = Eros(A, B, weights, similarity, dim_red_method)
		print("Eros distance:", distance)
	elif args.classify:
		weights = np.load(args.classify[3])
		similarity = args.classify[4]
		dim_red_method = args.classify[5]

		if similarity == 'euclidean':
			similarity = euclidean_distance
		elif similarity == 'cosine':
			similarity = cosine_similarity
		elif similarity == 'mse':
			similarity = mean_squared_error

		if dim_red_method == 'diffusion':
			dim_red_method = diffusion_map
		elif dim_red_method == 'pca':
			dim_red_method = pca

		result = classify_user(args.classify[0], args.classify[1], float(args.classify[2]), weights, similarity, dim_red_method)
		if result:
			print('User is classified as a member of the class.')
		else:
			print('User is not classified as a member of the class.')


if __name__ == "__main__":
	main()

# /Users/zoeynielsen/Desktop/Ethereum Research/address-filtering/Matrices2/0x0a11142eb9db99da8112001dc2c0d52e541b198d.csv