import numpy as np

from src.feature_map import feature_map, random_feature
from test_utils import check_arrays_close, check_scalars_close

class Test_feature_map:
    def test_0(self) -> None:
        """
        Tests that the correct thing is computed when point cloud is 
        random and charges are [1, 1]
        """
        random_vec = np.ones(3, dtype=np.float32)

        charges = np.array([1, 1])
        coords = np.random.normal(size=(2, 3)).astype(np.float32)

        coords_diff_0 = np.sum(coords[0] - coords[1])
        coords_diff_1 = np.sum(coords[1] - coords[0])

        expected_out = np.sin(coords_diff_0) + np.sin(coords_diff_1)

        out = feature_map(coords, charges, random_vec)

        check_scalars_close(out[0], expected_out)

        for i in range(1, 25):
            assert out[i] == 0



class Test_random_feature:
    def test_0(self) -> None:
        x = np.zeros(3, dtype=np.float32)
        y = np.zeros(3, dtype=np.float32)
        x[0] = 1.
        random_vector = np.zeros(3, dtype=np.float32)

        out = random_feature(x, y, random_vector)

        check_scalars_close(out, 0.)