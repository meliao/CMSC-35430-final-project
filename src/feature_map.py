import numpy as np
import numba

QM7_CHARGE_LST = [1., 6., 7., 8., 16.]
QM7_NUM_CHARGES = 5

@numba.jit(nopython=True)
def random_feature(x: np.ndarray,
                    y: np.ndarray,
                    random_vector: np.ndarray) -> np.float32:
    """
    Returns sin(<x - y, random_vector>)
    """

    return np.sin(np.dot(x - y, random_vector))

# @numba.jit(nopython=True)
def feature_map(coords: np.ndarray, 
                charges: np.ndarray,
                random_vector: np.ndarray) -> np.ndarray:
    """Given a sample defined by aligned coordinates and charges, compute a feature
    map defined by section 5 in the project writeup. Given charges c_1, c_2 the feature
    map is:
    
    sum_{x has charge c_1} sum_{y has charge c_2} sin(<x - y, random_vector>)

    We return a vector evaluating this feature map over all ordered pairs (c_1, c_2)

    Args:
        coords (np.ndarray): Has shape (n_atoms, 3)
        charges (np.ndarray): Has shape (n_atoms)
        random_vector (np.ndarray): Has shape (3)

    Returns:
        np.ndarray: _description_
    """
    out = np.zeros((QM7_NUM_CHARGES ** 2), dtype=np.float32)

    for idx_1, c_1 in enumerate(QM7_CHARGE_LST):
        c_1_bool_arr = charges == c_1
        c_1_coords = coords[c_1_bool_arr]
        for idx_2, c_2 in enumerate(QM7_CHARGE_LST):
            c_2_bool_arr = charges == c_2
            c_2_coords = coords[c_2_bool_arr]

            out_idx = idx_1 * QM7_NUM_CHARGES + idx_2

            for i in range(c_1_coords.shape[0]):
                x_i = c_1_coords[i]
                for j in range(c_2_coords.shape[0]):
                    y_j = c_2_coords[j]

                    out[out_idx] += random_feature(x_i, y_j, random_vector)

    return out


