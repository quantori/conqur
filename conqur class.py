import sklearn
import numpy


class ConQur(sklearn.base.TransformerMixin):
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
    the default value is equal to None, but the code will be executed on the list numpy.linspace(0.005, 1, 199).

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
        pass

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
        pass

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
