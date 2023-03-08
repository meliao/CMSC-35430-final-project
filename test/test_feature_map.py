import numpy as np

from src.feature_map import feature_map, random_feature
from test_utils import check_arrays_close, check_scalars_close

class Test_feature_map:
    def test_0(self) -> None:
        pass


class Test_random_feature:
    def test_0(self) -> None:
        x = np.zeros(3, dtype=np.float32)
        y = np.zeros(3, dtype=np.float32)
        x[0] = 1.
        random_vector = np.zeros(3, dtype=np.float32)

        out = random_feature(x, y, random_vector)

        check_scalars_close(out, 0.)