import sklearn as skl
import numpy as np
import pandas as pd


class ConQur(skl.base.TransformerMixin):
    """
    Parameters
    ----------
    batch_columns : The list of batch columns.

    covariates_columns : The list of covariates columns, which contains the key variable of interest and other
    covariates for each feature. None, if you want to drop the key variables and covariates, and then
    use the batch columns exclusively in regressions of both parts.

    penalty : See sklearn.linear_model.LogisticRegression; penalty parameter.

    dual : See sklearn.linear_model.LogisticRegression; dual parameter.

    tol : See sklearn.linear_model.LogisticRegression; tol parameter.

    C : See sklearn.linear_model.LogisticRegression; C parameter.

    fit_intercept_logit : The same as fit_intercept parameter, see sklearn.linear_model.LogisticRegression.

    intercept_scaling : See sklearn.linear_model.LogisticRegression; intercept_scaling parameter.

    class_weight : See sklearn.linear_model.LogisticRegression; class_weight parameter.

    random_state : See sklearn.linear_model.LogisticRegression; random_state parameter.

    solver_logit : The same as solver parameter, see sklearn.linear_model.LogisticRegression.

    max_iter : See sklearn.linear_model.LogisticRegression; max_iter parameter.

    multi_class : See sklearn.linear_model.LogisticRegression; multi_class parameter.

    verbose : See sklearn.linear_model.LogisticRegression; verbose parameter.

    warm_start : See sklearn.linear_model.LogisticRegression; warm_start parameter.

    n_jobs : See sklearn.linear_model.LogisticRegression; n_jobs parameter.

    l1_ratio : See sklearn.linear_model.LogisticRegression; l1_ratio parameter.

    alpha : See sklearn.linear_model.QuantileRegressor; alpha parameter.

    fit_intercept_quantile : The same as fit_intercept parameter, see sklearn.linear_model.QuantileRegressor.

    solver_quantile : The same as solver, see sklearn.linear_model.QuantileRegressor; penalty parameter.

    solver_options : See sklearn.linear_model.QuantileRegressor; solver_options parameter.

    quantiles : A sequence of quantile levels, determing the “precision” of estimating conditional quantile functions;
    the default value is equal to None, but the code will be executed on the list np.linspace(0.005, 1, 199).

    interplt_delta :  A float constant in (0, 0.5), determing the size of the interpolation window
    for using the data-driven linear interpolation between zero and non-zero quantiles to stabilize border estimates.
    None, if you don't use data-driven linear interpolation; default is None.


    """

    def __init__(self,
                 batch_columns,
                 covariates_columns,
                 *,
                 penalty='l2',
                 dual=False,
                 tol=1e-4,
                 C=1.0,
                 fit_intercept_logit=True,
                 intercept_scaling=1,
                 class_weight=None,
                 random_state=None,
                 solver_logit='lbfgs',
                 max_iter=100,
                 multi_class='auto',
                 verbose=0,
                 warm_start=False,
                 n_jobs=None,
                 l1_ratio=None,
                 alpha=1.0,
                 fit_intercept_quantile=True,
                 solver_quantile='interior-point',
                 solver_options=None,
                 quantiles=None,
                 interplt_delta=None
                 ):
        self.batch_columns = batch_columns
        self.covariates_columns = covariates_columns
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept_logit = fit_intercept_logit
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver_logit = solver_logit
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.fit_intercept_quantile = fit_intercept_quantile
        self.solver_quantile = solver_quantile
        self.solver_options = solver_options
        if quantiles == None:
            self.quantiles = np.linspace(0.005, 1, 199)
        else:
            self.quantiles = quantiles
        self.interplt_delta = interplt_delta

    def fit(self, X):
        """
        Parameters
        ----------
        X : Array-like of shape (n_samples, j_features)
            Training matrix, where `n_samples` is the number of samples for some i-th feature
            and `j_features` is the number of feature of this sample.

        Returns
        -------
        self
            Fitted scaler.


        """
        batch_and_covariates_indexes = np.hstack((self.batch_columns, self.covariates_columns))
        X_batch_and_covariates = X[:, batch_and_covariates_indexes]
        columns_indexes = np.arange(0, len(X[0]))
        feature_indexes = np.zeros(len(X[0]) - len(batch_and_covariates_indexes), dtype=np.int16)
        counter = 0
        for i in columns_indexes:
            if not (i in batch_and_covariates_indexes):
                feature_indexes[counter] = columns_indexes[i]
                counter += 1
        self.dict_logit_with_batch = dict()
        self.dict_quantile_with_batch = dict()
        for feature in feature_indexes:
            y_initial = X[:, feature]
            y_for_logit = y_initial.copy()
            y_for_logit[y_for_logit != 0] = 1
            logistic_regression = skl.linear_model.LogisticRegression(self.penalty,
                                                                      dual=self.dual,
                                                                      tol=self.tol,
                                                                      C=self.C,
                                                                      fit_intercept=self.fit_intercept_logit,
                                                                      intercept_scaling=self.intercept_scaling,
                                                                      class_weight=self.class_weight,
                                                                      random_state=self.random_state,
                                                                      solver=self.solver_logit,
                                                                      max_iter=self.max_iter,
                                                                      multi_class=self.multi_class,
                                                                      verbose=self.verbose,
                                                                      warm_start=self.warm_start,
                                                                      n_jobs=self.n_jobs,
                                                                      l1_ratio=self.l1_ratio
                                                                      )
            self.dict_logit_with_batch[feature] = logistic_regression.fit(X_batch_and_covariates, y_for_logit)
            y_nonzero = y_initial[y_initial != 0]
            y_nonzero_shifted = y_nonzero + np.random.uniform(0, 1, len(y_nonzero))
            X_and_y = np.vstack((X_batch_and_covariates, y_initial))
            X_and_y_nonzero = X_and_y[X_and_y[:, len(X_and_y[0]) - 1] != 0]
            X_batch_and_covariates_nonzero = X_and_y_nonzero[:, np.arange(0, len(X_and_y_nonzero[0]) - 1)]
            self.dict_quantile_with_batch[feature] = []
            for i in self.quantiles:
                quantile_regression = skl.linear_model.QuantileRegressor(quantile=i,
                                                                         alpha=self.alpha,
                                                                         fit_intercept=self.fit_intercept_quantile,
                                                                         solver=self.solver_quantile,
                                                                         solver_options=self.solver_options
                                                                         )
                self.dict_quantile_with_batch[feature].append(quantile_regression.fit(X_batch_and_covariates_nonzero,
                                                                                      y_nonzero_shifted))

    def transform(self, X):
        """
        Parameters
        ----------
        X : Array-like of shape  (n_samples, j_features)
            Input data that will be transformed.

        Returns
        -------
        Xt : Ndarray of shape (n_samples, j_features)
            Transformed data.


        """
        pass
