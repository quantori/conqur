class ConQur:
    """
Parameters
    ----------
    tax_tab : The taxa read count table, samples (row) by taxa (col).

    batchid : The batch indicator, must be a factor.

    covariates :  The data.frame contains the key variable of interest and other covariates, e.g.,
    data.frame(key, x1, x2).

    batch_ref : A character, the name of the reference batch, e.g.,“2”.

    logistic_lasso :  A logical value, TRUE for L1-penalized logistic regression, FALSE for standard
    logistic regression; default is FALSE.

    quantile_type :  A character, “standard” for standard quantile regression, “lasso” for L1-penalized
    quantile regression, “composite” for composite quantile regression; default is
    “standard”.

    simple_match : A logical value, TRUE for using the simple quantile-quantile matching, FALSE
    for not; default is FALSE.

    lambda_quantile : A character, the penalization parameter in quantile regression if quantile_type=“lasso”
    or “composite”; only two choices “2p/n” or “2p/logn”, where p is the number
    of expanded covariates and n is the number of non-zero read count; default is
    “2p/n”.

    interplt :  A logical value, TRUE for using the data-driven linear interpolation between
    zero and non-zero quantiles to stablize border estimates, FALSE for not; default
    is FALSE.

    delta A real constant in (0, 0.5), determing the size of the interpolation window if
    interplt=TRUE, a larger delta leads to a narrower interpolation window; default
    is 0.4999.

    taus : A sequence of quantile levels, determing the “precision” of estimating conditional quantile functions;
    default is [i for i in range(5, 1000, 5) / 1000].

    ----------




    """
    def __init__(self,
                tax_tab,
                batchid,
                covariates,
                batch_ref,
                logistic_lasso = False,
                quantile_type = "standard",
                simple_match = False,
                lambda_quantile = "2p/n",
                interplt = False,
                delta = 0.4999,
                taus = [i for i in range(5, 1000, 5) / 1000]
                ):
        pass
    def fit(self, X, y):
        pass
    def transform(self, X):
        pass
