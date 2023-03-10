#!/bin/bash

#SBATCH --job-name=2023-03-10_run_2_weight_stddev_100.0_n_features_1000_with_direction_T
#SBATCH --time=1:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=logs/2023-03-10/2023-03-10_run_2_weight_stddev_100.0_n_features_1000_with_direction_T.out
#SBATCH --error=logs/2023-03-10/2023-03-10_run_2_weight_stddev_100.0_n_features_1000_with_direction_T.err
### S B A T C H --exclude=

export NUMBA_NUM_THREADS=4
export OMP_NUM_THREADS=4

echo "`date` Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"
echo "    OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "    NUMBA_NUM_THREADS: ${NUMBA_NUM_THREADS}"


which python

python evaluate_model.py \
-data_fp data/qm7/qm7.mat \
-n_train 5732 \
-n_test 1433 \
-val_fraction 0.1 \
-with_directional_info T \
-save_results_fp data/results/2023-03-10_hyperparam_opt_3.txt \
-l2_reg 0. 1e-07 1e-06 1e-05 1e-04 1e-03 1e-02 1e-01 \
-n_features 1000 \
-random_vector_stddev 100.0 \
-save_data_fp data/generated/2023-03-10_run_2_weight_stddev_100.0_n_features_1000_with_direction_T.mat


