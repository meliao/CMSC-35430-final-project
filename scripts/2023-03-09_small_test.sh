date=2023-03-09
save_data_fp=data/generated/${date}_small_test.mat
save_results_fp=data/results/${date}_small_test.tex


python evaluate_model.py \
-data_fp data/qm7/qm7.mat \
-n_train 1_000 \
-n_test 1_000 \
-val_fraction 0.1 \
-n_features 100 \
-random_vector_stddev 1. \
-l2_reg 0 0.1 1e-02 1e-03 1e-04 1e-05 1e-06 \
-save_results_fp $save_results_fp \
-save_data_fp $save_data_fp \
-with_directional_info F