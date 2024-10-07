from sklearn.neighbors import NearestNeighbors
import numpy as np
import scipy.sparse

def find_neighbors(x: np.ndarray, n_neighbors: int = 2) -> scipy.sparse._lil.lil_matrix:
    """Function for constructing the symmetric nearest neighbors graph
    of rows in a data matrix and returning the corresponding graph Laplacian
    NOTE: This Laplacian is referred to as the "Symmetric Laplacian via the 
    incidence matrix" on Wikipedia.

    Parameters
    ----------
    x: np.ndarray
        Data matrix used to construct nearest neighbors graph
    n_neighbors: int
        Number of neighbors to consider

    Returns
    -------
    scipy.sparse._lil.lil_matrix
        Graph Laplacian
    """
    neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(x)

    kneighbors = neighbors.kneighbors(return_distance=False)
    pairs = np.vstack([
        np.repeat(np.arange(neighbors.n_samples_fit_), n_neighbors),
        kneighbors.flatten()])

    incidence = np.zeros((x.shape[0], pairs.shape[1]))
    for i, p in enumerate(pairs.T):
        incidence[p[0], i] = 1.
        incidence[p[1], i] = -1.
    laplacian = incidence @ incidence.T
    
    return pairs, laplacian