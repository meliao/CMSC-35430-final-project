from typing import List, Dict, Tuple
import logging

from scipy import linalg
import numpy as np

def _least_squares_soln(u: np.ndarray, s: np.ndarray, v_t: np.ndarray, labels: np.ndarray, l2_reg: float) -> np.ndarray:
    """
        w = (X^TX + lambda I)^{-1}X^Ty

        Using SVD: X = USV^T then we have 
        w = V(S^2 + lambda I)^{-1} S^T U^Ty

        where (S^2 + lambda I)^{-1} S^T =: D is a diagonal matrix with easy entries to compute
    Args:
        u, s, v_t (np.ndarray): Result of calling np.linalg.svd(feature_matrix, full_matrices=False)
        labels (np.ndarray): Labels that we are trying to fit
        l2_reg (float): Also known as lambda, the regularization parameter

    Returns:
        np.ndarray: One-dimension array of weights.
    """
    s_2 = np.square(s)

    # out = []
    # for reg in l2_reg:            
    pseudo_inv_denom = s_2 + l2_reg * np.ones_like(s)

    pseudo_inv_diag = np.divide(s, pseudo_inv_denom)
    pseudo_inv_diag = pseudo_inv_diag[:,np.newaxis]
    
    
    scaled_u_t = np.multiply(pseudo_inv_diag, u.transpose())
    A = np.matmul(v_t.transpose(), scaled_u_t)
    weights = np.matmul(A, labels).flatten()

    return weights

def svd_solve(feature_mat: np.ndarray, y: np.ndarray, l2_reg: List[float]) -> Tuple[Dict, np.ndarray]:
    """Finds ridge regression solutions to ||Ax - y||_2^2 + \\lambda || x||_2^2

    Args:
        feature_mat (np.ndarray): Has shape (n_samples, n_features)
        y (np.ndarray): Has shape (n_samples)
        l2_reg (List[float]): List of regularization lambda values

    Returns:
        Dict: Keys are the elements of l2_reg
    """
    logging.info("Beginning SVD-based ridge regression solve")
    u, s, v_t = linalg.svd(feature_mat, full_matrices=False)
    logging.info("SVD returned %i singular vals with range: [%f, %f] and cond number: %f", s.shape[0], s[0], s[-1], s[0] / s[-1])

    out_dd = {}
    for lambda_val in l2_reg:

        weights = _least_squares_soln(u, s, v_t, y, lambda_val)

        out_dd[lambda_val] = weights

    return out_dd, s