import numpy as np

from src.ridge_regression import svd_solve
from test_utils import check_arrays_close, check_scalars_close


class Test_svd_solve:
    def test_lstsq_random(self) -> None:
        """
        Tests least square solution computed on random normal matrix (50x10)
        """
        n = 50
        k = 10
        X = np.random.normal(size=(n,k))
        w = np.random.normal(size=(k,1))
        y = np.matmul(X, w)
        l = svd_solve(X, y, [0.,])[0][0.]
        w = w.flatten()
        check_arrays_close(l, w)

    def test_lstsq_ones(self) -> None:
        """
        Tests least squares solution on diagonal matrix (29x17)
        """
        n = 29
        k = 17
        X = np.eye(n,k)
        w = 4 * np.ones((k,1))
        y = np.matmul(X, w)
        lstsq_soln = svd_solve(X, y, [0.,])[0][0.]
        check_arrays_close(lstsq_soln, w)

    def test_lstsq_regularization(self) -> None:
        """
        Design matrix all ones, true weights all ones, nonzero l2 regularization coefficient
        """
        n = 7
        k = 3
        X = np.ones((n,k))
        w = np.ones((k,1))
        y = np.matmul(X,w)
        lambd = [1.]
        lstsq_soln = svd_solve(X, y, lambd)[0][1.]

        XtX = n * np.ones((k, k))
        part_a = np.linalg.inv(XtX + lambd * np.eye(k))
        part_b = np.matmul(X.transpose(), y)
        expected_ans = np.matmul(part_a, part_b)

        check_arrays_close(lstsq_soln, expected_ans, 'lstsq_soln', 'expected_ans')


    def test_lstsq_regularization_1(self) -> None:
        """
        A = I, y = e_1. Solution should give x_1 = 1 / (1 + lambda) and x_0 = 0
        """
        X = np.eye(2)
        y = np.array([1,0])
        lambd = [0., 1.1, 0.5]
        out = svd_solve(X, y, lambd)[0]
        for l in lambd:
            out_weights = out[l]
            expected_answer = np.array([1 / (1 + l), 0])
            check_arrays_close(out_weights, expected_answer, 'out_weights', 'expected_answer')
