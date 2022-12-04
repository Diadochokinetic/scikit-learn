import warnings
from numbers import Real, Integral
import numpy as np
from .base import BaseEstimator, RegressorMixin
from .compose import ColumnTransformer
from .preprocessing import KBinsDiscretizer
from .utils import check_array
from .utils._param_validation import Interval
from .utils.validation import check_is_fitted


class MarginalSumsRegression(BaseEstimator, RegressorMixin):
    """
    Marginal Sums Regression for aggregated and non aggreagted data.

    This is a multiplicative model based on a product of factors and a base value.
    f1 * f2 * ... * fn * b

    The base value is the mean of the target variable.
    There is a factor initialzed with 1 for each feature. The features are expected to
    be onehot encoded. The first column of X can be a weight vector (see add_weights).

    The features get updated sequentially. First the marginal sum for each feature is
    calculated by X.T dot y. These sums are fixed for the rest of the algorithm. For
    each feature X gets masked, such that all rows with the current feature = 1 are
    selected and all columns except for the current feature. This masked X gets
    multiplied elementwise with the corresponding weights and current factors of the
    other features.
    SUM(weights * PROD(factors) * y_mean)
    The original marginal sum for the current feature gets divided by this estimated
    marginal sum. The result is the updated factor
    marginal sum / estimated marginal sum

    This update is done iterativley until either a number of maximum iterations is
    reached or the algorithm converges.

    Parameters
    ----------
    discretizer : transformer, default="kbins"
        Transformer to discretize any numeric values in X, if they aren't onehot
        encoded yet. If "kbins", then dafault
        KBinsDiscretizer(encode="onehot", n_bins=2, random_state=42) is used.

    nax_iter : int, default=100
        Number of maximum iterations, in case the algorithm does not converge. One
        iteration consists of at least one factor update of each feature.

    min_factor_change : float, default=0.001
        Criteria for early stopping. Minimal change of at least one factor in the last
        iteration.

    Attributes
    ----------
    discretizer_ : transfoerm
        Internal discretizer used to onehot encode features.

    factors_ : ndarray of shape (n_classes,)
        Factors for the multiplicative model.

    factors_change_ : ndarray of shape (n_classes,)
        Updates on the factors of the latest iteration.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    weights_ : ndarray of shape (n_rows,)
        Weights used for fitting.

    y_mean_ : float
        Mean value of the target variable.
    """

    _parameter_constraints: dict = {
        "discretizer": "no_validation",
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "min_factor_change": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        discretizer="kbins",
        max_iter=100,
        min_factor_change=0.001,
    ):
        # self.add_weights = add_weights
        self.discretizer = discretizer
        self.max_iter = max_iter
        self.min_factor_change = min_factor_change

    def _check_X(self, X):
        # ensure ndarray
        if hasattr(X, "toarray"):
            X = X.toarray()

        # ensure array is onehot encoded
        onehot_cols = [
            col
            for col in range(X.shape[1])
            if ((X[:, col] == 0) | (X[:, col] == 1)).all()
        ]
        transform_cols = [
            col
            for col in range(X.shape[1])
            if not ((X[:, col] == 0) | (X[:, col] == 1)).all()
        ]
        if len(transform_cols) > 0:
            warnings.warn(
                f"Columns {transform_cols} are not onehot encoded and will be"
                " discretized.",
                UserWarning,
            )
            if self.discretizer == "kbins":
                self.discretizer_ = KBinsDiscretizer(
                    encode="onehot", n_bins=2, random_state=42
                )
            else:
                self.discretizer_ = self.discretizer

            self.column_transformer_ = ColumnTransformer(
                [
                    ("passthrough", "passthrough", onehot_cols),
                    ("discretizer", self.discretizer_, transform_cols),
                ]
            ).fit(X)
            X = self.column_transformer_.transform(X)
        else:
            self.column_transformer_ = None

        return X

    def _fit(self, X, y):
        """
        Fit the estimator by iterativly optimizing the factors for each feature.

        Parameters
        ----------

        X : array of shape (n,m)
            Input array with n observations and m features. All features need to be
            onehot encoded.

        y : array of shape (n,1)
            Target variable.
        """

        for i in range(1, self.max_iter + 1):
            self.n_iter_ = i
            for feature in range(X.shape[1]):
                # Create a mask to select all rows with the current feature = 1 and
                # all columns except for the current feature.
                col_mask = [True if i != feature else False for i in range(X.shape[1])]
                row_mask = X[:, feature] > 0

                # Mask X and multiply elementwise with the current factors
                X_factor = np.multiply(
                    self.factors_[col_mask], X[row_mask][:, col_mask]
                )

                # Calculate the marginal sum with the current factors
                # SUM(weights * PROD(factors) * y_mean)
                calc_marginal_sum = (
                    self.weights_[row_mask]
                    * np.prod(X_factor, axis=1, where=X_factor > 0)
                    * self.y_mean_
                ).sum()

                # Update the factor
                updated_factor = self.marginal_sums_[feature] / calc_marginal_sum
                self.factors_change_[feature] = np.absolute(
                    self.factors_[feature] - updated_factor
                )
                self.factors_[feature] = updated_factor

            # Check early stopping criteria after each iteration
            if np.max(self.factors_change_) < self.min_factor_change:
                print(f"Converged after {self.n_iter_} iterations.")
                break

            if i == self.max_iter:
                warnings.warn(
                    f"Did not converge after {self.max_iter} iterations.", UserWarning
                )

    def fit(self, X, y, sample_weight=None):
        """
        Wrapper for the fit_ method to calculated weights, marginal sums, the mean
        target and initialize factors.

        Parameters
        ----------
        X : array of shape (n,m)
            Input array with n observations and either m features (no weight vector) or
            m - 1 features if the first row is a weight vector. All features, except
            for the weight vector, need to be onehot encoded. The weights must not
            contain zeros.

        y : array of shape (n,1)
            Target variable.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).
        """

        self._validate_params()
        X, y = self._validate_data(X=X, y=y, reset=True, accept_sparse=True)
        X = self._check_X(X)

        # init weight vector
        if sample_weight is not None:
            self.weights_ = sample_weight
        else:
            self.weights_ = np.ones(X.shape[0])
        if (self.weights_ < 0).any():
            raise ValueError(
                "Value < 0 detected in first column. Expected weights >= 0."
            )

        # init factors
        self.factors_ = np.ones(X.shape[1])
        self.factors_change_ = np.zeros(X.shape[1])

        # calculate marginal sums of original data
        self.marginal_sums_ = np.dot(X.T, y)

        # calculate mean y
        self.y_mean_ = np.sum(y) / np.sum(self.weights_)

        self._fit(X, y)

    def predict(self, X):
        """
        Predict based on the fitted model. This method expects no weight vector, only
        the onehot encoded features.

        Parameters
        -----------

        X : array of shape (n,m)
            Input array with n observations and m features. All features need to be
            onehot encoded.
        """
        check_is_fitted(self)
        X = self._validate_data(X=X, accept_sparse=True, reset=False)
        if self.column_transformer_:
            X = self.column_transformer_.transform(X)
        X_factor = np.multiply(self.factors_, X)
        return np.prod(X_factor, axis=1, where=X_factor > 0) * self.y_mean_

    def fit_predict(self, X, y, sample_weight=None):
        """
        Fit & predict. See the fit and predcit methods for details.

        Parameters
        ----------
        X : array of shape (n,m)
            Input array with n observations and either m features (no weight vector) or
            m - 1 features if the first row is a weight vector. All features, except
            for the weight vector, need to be onehot encoded. The weights must not
            contain zeros.

        y : array of shape (n,1)
            Target variable.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).
        """

        self.fit(X, y, sample_weight)
        return self.predict(X)
