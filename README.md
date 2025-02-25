# Address Filtering

This repository currently contains functions to compare address activity on Ethereum given some dataset generated in format given by [ethereum-etl](https://github.com/blockchain-etl/ethereum-etl) and a collection of addresses under the same class using methods outlined by [Defining user spectra to classify Ethereum users based on their behavior](https://link.springer.com/content/pdf/10.1186/s40537-022-00586-3.pdf).

## Functionality
- **Matrix Representation**: Given a dataset with Ethereum transaction data and a dataset of addresses of interest, it extracts the following values for each given address in each block: out_tx, in_tx, out_value, in_value, unique_receivers, and unique_senders.
- **Eros Distance**: Calculates the [Eros Distance](https://static.aminer.org/pdf/PDF/000/504/527/a_pca_based_similarity_measure_for_multivariate_time_series.pdf) between two matrices constructed based on the transaction data. This distance, while traditionally based in PCA, can be modified in this program to work with diffusion maps and use different similarity metrics.
- **Comparison**: Compare how two different versions of the Eros distance may perform. This is done by calculating the difference between addresses of the same class and addresses that are not a part of the class, taking the average of the giving values, and outputting the mean and standard deviation. These values can be used to compare and determine what Eros distance is optimal.

## Functions
#### `compare_distances(folder1, folder2, similarity, dim_red_1, dim_red_2, max_iter=100):`
- `folder1`: folder containing the matrices of the class of interest
- `folder2`: folder containing matrices not in the class of interest
- `similarity`: similarity metric used in Eros definition. Input is two vectors of same dimension and output is scalar value
- `dim_red_1`: dimensionality reduction technique that takes a matrix as input and outputs eigenvalues and eigenvectors
- `dim_red_2`: dimensionality reduction technique that takes a matrix as input and outputs eigenvalues and eigenvectors
- `max_iter`: maximum number of matrices in `folder1` and `folder2` to look at

**Outputs:**
- Print summary with mean and standard deviation comparing the two methods.


#### `construct_matrix(path, blacklist)`
- `path` (str): Path to the Ethereum transaction CSV dataset.
- `blacklist` (str): Path to the CSV file containing a list of addresses of interest.

**Outputs:**
- CSV files: `interesting_senders.csv` and `interesting_receivers.csv` containing the filtered addresses.
- Matrices: CSV files for each address in the `Matrices/` directory containing the transaction data.


#### `construct_matrix_not_interest(path, blacklist)`
- `path` (str): Path to the Ethereum transaction CSV dataset.
- `blacklist` (str): Path to the CSV file containing a list of addresses of interest.

**Outputs:**
- Matrices: CSV files for each address in the `CompareMatrices/` directory containing the transaction data that was not in our addresses of interest list
