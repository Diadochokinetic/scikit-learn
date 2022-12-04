import numpy as np
import pytest
import scipy
from sklearn.exceptions import DataConversionWarning
from sklearn.marginal_totals import MarginalTotalsRegression
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import SkipTest

weights = np.array([300, 700, 600, 200])
y = np.array([220, 330, 200, 300])
X = np.array(
    [
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
    ]
)
factors = np.array([1.04907574, 0.95345681, 1.18709246, 0.79148547])
y_pred = [220.03697532, 330.01772326, 199.98151234, 299.9379686]


def test_with_given_weights():
    msr = MarginalTotalsRegression()
    msr.fit(X, y, sample_weight=weights)

    assert_array_almost_equal(msr.factors_, factors)
    assert_array_almost_equal(msr.predict(X), y_pred)


def test_without_weights():
    msr = MarginalTotalsRegression()
    msr.fit(np.repeat(X, weights, axis=0), np.repeat(y, weights, axis=0))

    assert_array_almost_equal(msr.factors_, factors)
    assert_array_almost_equal(msr.predict(X), y_pred)


def test_negative_weights():
    neg_weights = np.array([-100, 200, 300, 400])

    msr = MarginalTotalsRegression()
    msg = r"Value < 0 detected in first column. Expected weights >= 0."
    with pytest.raises(ValueError, match=msg):
        msr.fit(X, y, sample_weight=neg_weights)


def test_convergence_warning():
    msr = MarginalTotalsRegression(max_iter=3)

    msg = r"not converge"
    with pytest.warns(UserWarning, match=msg):
        msr.fit(X, y, sample_weight=weights)


def test_sparse_input():
    msr = MarginalTotalsRegression()

    msg = r"A sparse matrix was passed"
    with pytest.raises(TypeError, match=msg):
        msr.fit(scipy.sparse.csr_matrix(X), y, sample_weight=weights)


def test_not_onehot_encoded_input():
    X = np.array(
        [
            [2, 0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [3, 0.0, 0.0, 1.0],
            [1, 0.0, 1.0, 0.0],
        ]
    )

    msr = MarginalTotalsRegression()
    msg = r"are not onehot encoded and will be discretized."
    with pytest.warns(UserWarning, match=msg):
        msr.fit(X, y, sample_weight=weights)


def test_2d_y():
    msr = MarginalTotalsRegression()
    msg = r"A column-vector y was passed when a 1d array was expected."
    with pytest.warns(DataConversionWarning, match=msg):
        msr.fit(X, y.reshape(-1, 1), sample_weight=weights)

    assert_array_almost_equal(msr.factors_, factors)
    assert_array_almost_equal(msr.predict(X), y_pred)


def test_df_with_given_weights():

    try:
        import pandas as pd
    except ImportError:
        raise SkipTest(
            "pandas is not installed: not checking column name consistency for pandas"
        )
    msr = MarginalTotalsRegression()

    columns = ["x1", "x2", "x3", "x4"]
    msr.fit(pd.DataFrame(X, columns=columns), y, sample_weight=weights)

    assert_array_almost_equal(msr.factors_, factors)
    assert_array_almost_equal(msr.predict(pd.DataFrame(X, columns=columns)), y_pred)
