import logging
import argparse

import numpy as np

from src.feature_map import feature_map
from src.QM7Dataset import load_QM7


def setup_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument('-data_fp', help='Where QM7 data is stored')
    parser.add_argument('-n_features', type=int)
    parser.add_argument('-random_vector_stddev', type=float)