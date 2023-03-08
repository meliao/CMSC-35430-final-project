import numpy as np

from src.QM7Dataset import _align_coords
from test_utils import check_arrays_close, check_scalars_close

class Test__align_coords:
    def test_0(self) -> None:
        """
        Direction of max variation is first coordinate, direction of 2nd max variation is 2nd coordinate, ...
        so the alignment shouldn't change the points at all.
        """
        point_cloud = np.zeros((3, 4))
        point_cloud[0, 0] = -10.
        point_cloud[1, 1] = 3.
        point_cloud[2, 2] = 1.
        point_cloud[0, 3] = 10.

        out_aligned = _align_coords(point_cloud)

        assert point_cloud.shape == out_aligned.shape

        check_arrays_close(point_cloud, out_aligned)

    def test_1(self) -> None:
        """
        Tests that randomly-generated inputs dont break anything
        """
        np.random.seed(1000)

        for i in range(10):
            point_cloud = np.random.normal(size=(3, 12))

            out_aligned = _align_coords(point_cloud)

