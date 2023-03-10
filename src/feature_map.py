import numpy as np
import numba

QM7_NUM_CHARGES = 5

@numba.jit(nopython=True)
def random_feature_with_directional_info(x: np.ndarray,
                                            y: np.ndarray,
                                            random_vector: np.ndarray) -> np.float32:
    """
    Returns sin(<x - y, random_vector> || x - y||^{-2})
    """
    xy_norm = np.linalg.norm(x - y)
    xy_normalized = (x - y) / xy_norm

    return np.sin(np.dot(xy_normalized, random_vector) * (xy_norm ** -2))

@numba.jit(nopython=True)
def random_feature_without_directional_info(x: np.ndarray,
                                            y: np.ndarray,
                                            random_vector: np.ndarray) -> np.float32:
    """
    Returns sin( ||x - y||^{-2} * random_vector)
    """
    xy_norm = np.linalg.norm(x - y)

    return np.sin(random_vector[0] * (xy_norm ** -2))

@numba.jit(nopython=True)
def feature_map(coords: np.ndarray, 
                charges: np.ndarray,
                random_vector: np.ndarray,
                with_directional_info: bool) -> np.ndarray:
    """Given a sample defined by aligned coordinates and charges, compute a feature
    map defined by section 5 in the project writeup. Given charges c_1, c_2 the feature
    map is:
    
    sum_{x has charge c_1} sum_{y has charge c_2} random_feature(x, y, random_vector)

    We return a vector evaluating this feature map over all ordered pairs (c_1, c_2)

    Args:
        coords (np.ndarray): Has shape (n_atoms, 3)
        charges (np.ndarray): Has shape (n_atoms)
        random_vector (np.ndarray): Has shape (3)
        with_directional_info (bool): Whether to use directional information when computing the random feature

    Returns:
        np.ndarray: _description_
    """
    # Needs to be inside function def to make numba compile
    QM7_CHARGE_LST = [1., 6., 7., 8., 16.]


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
                    if c_1 == c_2 and i == j:
                        continue
                    y_j = c_2_coords[j]

                    if with_directional_info:
                        out[out_idx] += random_feature_with_directional_info(x_i, y_j, random_vector)
                    else:
                        out[out_idx] += random_feature_without_directional_info(x_i, y_j, random_vector)
                        

    return out


@numba.jit(nopython=True, parallel=True)
def feature_map_on_dset(coords: np.ndarray,
                        charges: np.ndarray,
                        n_atoms: np.ndarray,
                        random_weights: np.ndarray,
                        with_directional_info: bool) -> np.ndarray:
    """Computes a feature matrix where entry i,j is the jth random feature evaluated on sample i.

    Args:
        coords (np.ndarray): shape (n_samples, max_n_atoms, 3)
        charges (np.ndarray): shape (n_samples, max_n_atoms)
        n_atoms (np.ndarray): shape (n_samples,)
        random_weights (np.ndarray): shape (n_features, 3)

    Returns:
        np.ndarray: shape (n_samples, n_features * QM7_NUM_CHARGES * QM7_NUM_CHARGES) = (n_samples, n_features * 25)
    """
    
    n_features = random_weights.shape[0]
    n_samples = coords.shape[0]

    out = np.full((n_samples, n_features * QM7_NUM_CHARGES * QM7_NUM_CHARGES), np.nan, np.float32)
    for i in numba.prange(n_samples):
        n_atoms_i = n_atoms[i]
        charges_i = charges[i, :n_atoms_i]
        coords_i = coords[i, :n_atoms_i]

        for j in range(n_features):
            idx_start = j * QM7_NUM_CHARGES * QM7_NUM_CHARGES
            idx_end = (j + 1) * QM7_NUM_CHARGES * QM7_NUM_CHARGES
            out[i, idx_start:idx_end] = feature_map(coords_i, charges_i, random_weights[j], with_directional_info)

    return out


