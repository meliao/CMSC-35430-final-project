import logging
from typing import Tuple, Type, List

from scipy import io
import numpy as np

KCAL_MOL_TO_EV = 0.0433634
class QM7Data:
    def __init__(self, 
                    coords: np.ndarray,
                    charges: np.ndarray,
                    atomization_energies: np.ndarray) -> None:
        """
        coords has shape (n_samples, max_n_atoms, 3)
        charges has shape (n_samples, max_n_atoms)
        atomization_energies has shape (n_samples,)
        """
        self.coords = coords
        self.charges = charges
        self.atomization_energies = atomization_energies

        self.n_atoms = np.sum(self.charges != 0, axis=1)

        self.n_samples, self.max_n_atoms = self.charges.shape

        self.aligned_coords = None

    def __len__(self) -> int:
        return self.n_samples

    def align_coords(self: None) -> None:
        """Uses PCA to align molecules into a canonical alignment.

        Let X have shape (3, n_atoms) and let it be the Cartesian coords of the molecular point cloud. 
        Then X = USV^t is its SVD and we want to use U^tX as our aligned coordinates.
        """
        self.aligned_coords = np.zeros_like(self.coords)
        for i in range(self.n_samples):
            n_atoms_i = self.n_atoms[i]

            coords_i = self.coords[i, :n_atoms_i]
            
            self.aligned_coords[i, :n_atoms_i] = _align_coords(coords_i.T).T

def _align_coords(point_cloud: np.ndarray) -> np.ndarray:
    """
    Expects input of size (3, n_points). 
    If X is the input and X = USV^t is the SVD, this function returns U^tX. 
    The rows of U are the singular vectors. We want U[0,0] > 0 and U[1, 0] > 0 to 
    alleviate any sign ambiguity. 
    """
    u, _, _ = np.linalg.svd(point_cloud, full_matrices=False, compute_uv=True)

    if u[0, 0] >= 0 and u[1, 0] >= 0:
        sign_mat = np.eye(3)
    elif u[0, 0] >= 0 and u[1, 0] < 0:
        sign_mat = np.diag([1., -1., -1.])
    elif u[0, 0] < 0 and u[1, 0] >= 0:
        sign_mat = np.diag([-1., 1., -1.])
    elif u[0, 0] < 0 and u[1, 0] < 0:
        sign_mat = np.diag([-1., -1., 1.])
    else:
        raise ValueError(f"Can't figure out sign pattern for {u}")


    signed_u = np.matmul(sign_mat, u)
    return np.matmul(signed_u.T, point_cloud)

def load_QM7(fp: str, 
                n_train: int=2, 
                n_test: int=2,
                validation_set_fraction: float=0.1,
                permute_samples: bool=True) -> Tuple[QM7Data, QM7Data, QM7Data]:
    """Loads the QM7 dataset but NEVER loads the final fold (which will be 
    used as a held-out validation set). 

    Args:
        fp (str): path to the QM7 file
        n_train_folds (int, optional): Number of folds (each of size 1433) to be included in the train dataset. Defaults to 2.
        n_test_folds (int, optional): Number of folds (each of size 1433) to be included in the test dataset. Defaults to 2.

    Returns:
        Tuple[ElementPairsDatasetEncoding, ElementPairsDatasetEncoding]: train data and test data
    """

    data = io.loadmat(fp)

    # Atomization energies (labels)
    T = data['T'].flatten() * KCAL_MOL_TO_EV # now units are in eV
    # Splits for CV
    P = data['P']
    # Cartesian coordinates of atoms (size 7165 x 23 x 3)
    R = data['R']
    # Charges of atoms (size 7165 x 23)
    Z = data['Z']

    n_validation = int(np.floor(n_train * validation_set_fraction))
    n_train_eff = n_train - n_validation

    assert n_train + n_test <= T.shape[0]

    logging.info("Releasing train, validation, test datasets of size %i, %i, %i", n_train_eff, n_validation, n_test)

    if permute_samples:
        perm = np.random.permutation(T.shape[0])
    else:
        perm = np.arange(T.shape[0])
    train_idxes = perm[:n_train_eff]
    val_idxes = perm[n_train_eff:n_train]
    test_idxes = perm[n_train:n_train + n_test]

    assert np.intersect1d(train_idxes, val_idxes).shape[0] == 0
    assert np.intersect1d(train_idxes, test_idxes).shape[0] == 0
    assert np.intersect1d(test_idxes, val_idxes).shape[0] == 0

    train_dset = QM7Data(coords=R[train_idxes],
                            charges=Z[train_idxes],
                            atomization_energies=T[train_idxes])
    val_dset = QM7Data(coords=R[val_idxes],
                            charges=Z[val_idxes],
                            atomization_energies=T[val_idxes])

    test_dset = QM7Data(coords=R[test_idxes],
                            charges=Z[test_idxes],
                            atomization_energies=T[test_idxes])

    return (train_dset, val_dset, test_dset)