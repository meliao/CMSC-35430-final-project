from typing import Tuple
import logging
import argparse

import numpy as np
from scipy import io

from src.feature_map import feature_map_on_dset
from src.QM7Dataset import load_QM7
from src.ridge_regression import svd_solve
from src.logging_utils import write_result_to_file, FMT, TIMEFMT


def setup_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument('-data_fp', help='Where QM7 data is stored')
    parser.add_argument('-n_train', type=int)
    parser.add_argument('-val_fraction', type=float, default=0.1)
    parser.add_argument('-n_test', type=int)
    parser.add_argument('-n_features', type=int)
    parser.add_argument('-random_vector_stddev', type=float)
    parser.add_argument('-l2_reg', type=float, nargs='+', default=[0.])
    parser.add_argument('-save_results_fp')
    parser.add_argument('-save_data_fp')

    return parser.parse_args()

def predict_and_compute_error(features: np.ndarray, weights: np.ndarray, true_labels: np.ndarray) -> Tuple:
    """_summary_

    Args:
        features (np.ndarray): Has shape (n, d)
        weights (np.ndarray): Has shape (d)
        true_labels (np.ndarray): Has shape (n)

    Returns:
        Tuple: predictions (shape (n)), MAE
    """
    preds = np.matmul(features, weights.flatten()).flatten()

    MAE = np.mean(np.abs(preds.flatten() - true_labels.flatten()))

    return preds, MAE


def main(args: argparse.Namespace) -> None:
    """
    1. load data
    2. split into train/val/test
    3. draw random weights
    4. evaluate random feature matrices
    5. train linear weights
    6. evaluate performance and save
    """
    # Load data and split into train/val/test
    train_dset, val_dset, test_dset = load_QM7(args.data_fp, 
                                                args.n_train, 
                                                args.n_test, 
                                                args.val_fraction)
    n_train = len(train_dset)
    n_val = len(val_dset)
    n_test = len(test_dset)
    logging.info("Loaded %i train samples, %i val samples, %i test samples", n_train, n_val, n_test)
    
    # Align the molecules
    logging.info("Aligning the point clouds")
    train_dset.align_coords()
    val_dset.align_coords()
    test_dset.align_coords()
    
    # Draw random weights
    random_weights_shape = (args.n_features, 3)
    logging.info("Drawing random weights of shape %s and stddev %f", random_weights_shape, args.random_vector_stddev)
    random_weights = np.random.normal(size=random_weights_shape, scale=args.random_vector_stddev).astype(np.float32)

    # Compute random feature matrices
    logging.info("Computing random features on the train set")
    train_random_features = feature_map_on_dset(train_dset.aligned_coords,
                                                train_dset.charges,
                                                train_dset.n_atoms,
                                                random_weights)

    logging.info("Computing random features on the val set")
    val_random_features = feature_map_on_dset(val_dset.aligned_coords,
                                                val_dset.charges,
                                                val_dset.n_atoms,
                                                random_weights)

    logging.info("Computing random features on the test set")
    test_random_features = feature_map_on_dset(test_dset.aligned_coords,
                                                test_dset.charges,
                                                test_dset.n_atoms,
                                                random_weights)


    # Train linear weights
    weight_dd, s = svd_solve(train_random_features, train_dset.atomization_energies, args.l2_reg)

    train_pred_lst = []
    val_pred_lst = []
    test_pred_lst = []
    weight_lst = []
    for l2_reg, weights in weight_dd.items():

        train_preds, train_MAE = predict_and_compute_error(train_random_features, weights, train_dset.atomization_energies)
        val_preds, val_MAE = predict_and_compute_error(val_random_features, weights, val_dset.atomization_energies)
        test_preds, test_MAE = predict_and_compute_error(test_random_features, weights, test_dset.atomization_energies)

        logging.info("L2 reg: %f. Train MAE: %f. Val MAE: %f. Test MAE: %f", l2_reg, train_MAE, val_MAE, test_MAE)

        train_pred_lst.append(train_preds)
        val_pred_lst.append(val_preds)
        test_pred_lst.append(test_preds)
        weight_lst.append(weights)


        experiment_dd = {
                # Experiment hyperparameters
                'n_train': n_train,
                'n_val': n_val, 
                'n_test': n_test, 
                'n_features': args.n_features,
                'random_weights_scale': args.random_vector_stddev,
                'l2_reg': l2_reg,

                # Supplementary data
                'min_singular_val': s[-1],
                'max_singular_val': s[0],
                
                # Performance Results
                'train_MAE': train_MAE,
                'val_MAE': val_MAE,
                'test_MAE': test_MAE,
            }
        write_result_to_file(args.save_results_fp, **experiment_dd)


    out_data_dd = {
        'train_features_singular_values': s,
        'train_predictions': np.stack(train_pred_lst),
        'val_predictions': np.stack(val_pred_lst),
        'test_predictions': np.stack(test_pred_lst),
        'l2_reg_values': np.array(args.l2_reg),
        'train_labels': train_dset.atomization_energies,
        'val_labels': val_dset.atomization_energies,
        'test_labels': test_dset.atomization_energies,
        'random_weights': random_weights,
        'trained_weights': np.stack(weight_lst),
    }
    io.savemat(args.save_data_fp, out_data_dd)






if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format=FMT,
                        datefmt=TIMEFMT)
    args = setup_args()

    main(args)