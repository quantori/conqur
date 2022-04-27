import sklearn
import numpy as nm

class ConQur(sklearn.base.TransformerMixin):
    """
    Parameters
    ----------
    tax_tab : The taxa read count table, samples (row) by taxa (col).

    batchid : The batch indicator.

    batch_ref : A character, the name of the reference batch, e.g.,“2”.

    covariates :  The data.frame contains the key variable of interest and other covariates, e.g.,
    data.frame(key, x1, x2). None, if you want to drop the key variables and covariates, and then use the batch ID
    exclusively in regressions of both parts; default is None.

    logistic_lasso :  A logical value, TRUE for L1-penalized logistic regression, FALSE for standard
    logistic regression; default is FALSE.

    l1_coefficient : A value for L1-penalized quantile regression; default is 0.

    number_of_composites : The number of composites in composite quantile regression; default is 1.

    lambda_quantile : A real number, the penalization parameter in quantile regression if used lasso or composite
    quantile regression; default is 1.

    interplt_delta :  A real constant in (0, 0.5), determing the size of the interpolation window
    for using the data-driven linear interpolation between zero and non-zero quantiles to stablize border estimates.
    None, if you don't use data-driven linear interpolation; default is None.

    taus : A sequence of quantile levels, determing the “precision” of estimating conditional quantile functions;
    default is nm.arange(0.005, 1, 0.005).

    ----------




    """
    def __init__(self,
                tax_tab,
                batchid,
                batch_ref,
                covariates = None,
                logistic_lasso = False,
                l1_coefficient = 0,
                number_of_composites = 1,
                lambda_quantile = 1,
                interplt_delta = None,
                taus = nm.arange(0.005, 1, 0.005)
                ):
        pass
    def fit(self, X, y):
        pass
    def transform(self, X):
        pass
