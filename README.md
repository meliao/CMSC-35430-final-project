# CMSC-35430-final-project

## Testing
The following command runs all tests:
```
python -m pytest test/
```

## Usage Example
```
python evaluate_model.py \
-data_fp data/qm7/qm7.mat \
-n_train 5732 \
-n_test 1433 \
-val_fraction 0.1 \
-with_directional_info F \
-save_results_fp data/results/2023-03-10_hyperparam_opt_2.txt \
-l2_reg 0. 1e-07 1e-06 1e-05 1e-04 1e-03 1e-02 1e-01 \
-n_features 100 \
-random_vector_stddev 0.1 \
-save_data_fp data/generated/2023-03-10_run_2_weight_stddev_0.1_n_features_100_with_direction_False.mat
```

## Hyperparam Optimization Jobs
They're all stored in the folder `jobs/2023-03-10`. They were generated using files in `scripts/dsi_cluster` using an automated job submission tool. These jobs must be executed from the root directory of this project. 