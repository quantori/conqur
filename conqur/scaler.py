import sklearn.linear_model
import numpy as np


class ConQur(sklearn.base.TransformerMixin):
    """
    Parameters
    ----------
    batch_columns : A list of batch columns.

    covariates_columns : A list of covariates columns, which contains the key variable of interest and other
    covariates for each feature. None, if you want to drop the key variables and covariates, and then
    use the batch columns exclusively in regressions of both parts.

    jittering_columns : A list of columns whose elements are integers. Such columns are understood as a
    discrete distribution, and they need to be shifted to a uniform distribution. Must not intersect with
    covariates_columns and batch_columns. If None, then all columns of features are shifted; default is None.

    reference_batch : An integer constant indicating the control package. Must be a batch_columns element;
    default is 0.

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

    alphas : A list of L1-regularization constants, the size of the list must be equal to the number
    of feature columns. The default value is equal to None, but the code will be executed on the list
    np.array(1.0, ..., 1.0).

    fit_intercept_quantile : The same as fit_intercept parameter, see sklearn.linear_model.QuantileRegressor.

    solver_quantile : The same as solver, see sklearn.linear_model.QuantileRegressor; penalty parameter.

    solver_options : See sklearn.linear_model.QuantileRegressor; solver_options parameter.

    quantiles : A sequence of quantile levels, determing the “precision” of estimating conditional quantile functions;
    the default value is equal to None, but the code will be executed on the list
    np.linspace(0.05, 1, 19, endpoint=False).

    interplt_delta : A float constant in (0, 0.5), determing the size of the interpolation window
    for using the data-driven linear interpolation between zero and non-zero quantiles to stabilize border estimates.
    None, if you don't use data-driven linear interpolation; default is None.


    """

    def __init__(self,
                 batch_columns,
                 covariates_columns,
                 *,
                 jittering_columns=None,
                 reference_batch=0,
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
                 alphas=None,
                 fit_intercept_quantile=True,
                 solver_quantile='interior-point',
                 solver_options=None,
                 quantiles=None,
                 interplt_delta=None
                 ):
        self.batch_columns = batch_columns
        self.covariates_columns = covariates_columns
        self.jittering_columns = jittering_columns
        self.reference_batch = reference_batch
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
        self.alphas = alphas
        self.fit_intercept_quantile = fit_intercept_quantile
        self.solver_quantile = solver_quantile
        self.solver_options = solver_options
        if quantiles is None:
            self.quantiles = np.linspace(0.05, 1, 19, endpoint=False)
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
        batch_columns_no_ref_batch = list(set(self.batch_columns) - set(np.array([self.reference_batch])))
        batch_columns_no_ref_batch = np.array(batch_columns_no_ref_batch)
        batch_and_covariates_no_ref_batch_indexes = np.hstack((batch_columns_no_ref_batch, self.covariates_columns))
        X_batch_and_covariates_no_ref_batch = X[:, batch_and_covariates_no_ref_batch_indexes]
        columns_indexes_set = set(np.arange(0, len(X[0])))
        batch_and_covariates_indexes_set = set(batch_and_covariates_indexes)
        feature_indexes = list(columns_indexes_set - batch_and_covariates_indexes_set)
        feature_indexes = np.array(feature_indexes)
        if self.jittering_columns is None:
            jittering_columns_indexes = feature_indexes
        else:
            jittering_columns_indexes = self.jittering_columns
        if self.alphas is None:
            alphas_list = np.ones(len(feature_indexes))
        else:
            alphas_list = self.alphas
        self.dict_logit_with_batch = dict()
        self.dict_quantile_with_batch = dict()
        for feature in feature_indexes:
            y_initial = X[:, feature]
            y_for_logit = y_initial.copy()
            y_for_logit[y_for_logit != 0] = 1
            if all(y_for_logit == np.ones(len(y_for_logit))):
                self.dict_logit_with_batch[feature] = (1.0, 'all_positive')
            elif all(y_for_logit == np.zeros(len(y_for_logit))):
                self.dict_logit_with_batch[feature] = (0.0, 'all_equal_to_zero')
                continue
            else:
                logistic_regression = sklearn.linear_model.LogisticRegression(self.penalty,
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
                self.dict_logit_with_batch[feature] = (logistic_regression.fit(X_batch_and_covariates_no_ref_batch,
                                                                               y_for_logit), 'model')
            y_nonzero = y_initial[y_initial != 0]
            y_nonzero_shifted = y_nonzero
            if feature in jittering_columns_indexes:
                y_nonzero_shifted = y_nonzero + np.random.uniform(0, 1, len(y_nonzero))
            X_and_y = np.hstack((X_batch_and_covariates_no_ref_batch, y_initial.reshape(len(y_initial), 1)))
            X_and_y_nonzero = X_and_y[X_and_y[:, len(X_and_y[0]) - 1] != 0]
            X_batch_and_covariates_nonzero = X_and_y_nonzero[:, :-1]
            self.dict_quantile_with_batch[feature] = []
            for i in self.quantiles:
                quantile_regression = sklearn.linear_model.QuantileRegressor(quantile=i,
                                                                             alpha=alphas_list[feature],
                                                                             fit_intercept=self.fit_intercept_quantile,
                                                                             solver=self.solver_quantile,
                                                                             solver_options=self.solver_options
                                                                             )
                self.dict_quantile_with_batch[feature].append(quantile_regression.fit(X_batch_and_covariates_nonzero,
                                                                                      y_nonzero_shifted))
        return self

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
        Xt = X.copy()
        batch_and_covariates_indexes = np.hstack((self.batch_columns, self.covariates_columns))
        batch_columns_no_ref_batch = list(set(self.batch_columns) - set(np.array([self.reference_batch])))
        batch_columns_no_ref_batch = np.array(batch_columns_no_ref_batch)
        batch_and_covariates_no_ref_batch_indexes = np.hstack((batch_columns_no_ref_batch, self.covariates_columns))
        X_batch_and_covariates_no_ref_batch = X[:, batch_and_covariates_no_ref_batch_indexes]
        columns_indexes_set = set(np.arange(0, len(X[0])))
        batch_and_covariates_indexes_set = set(batch_and_covariates_indexes)
        feature_indexes = list(columns_indexes_set - batch_and_covariates_indexes_set)
        feature_indexes = np.array(feature_indexes)
        if self.jittering_columns is None:
            jittering_columns_indexes = feature_indexes
        else:
            jittering_columns_indexes = self.jittering_columns
        if self.alphas is None:
            alphas_list = np.ones(len(feature_indexes))
        else:
            alphas_list = self.alphas
        for feature in feature_indexes:
            y_initial = X[:, feature]
            if self.dict_logit_with_batch[feature][1] == 'all_positive':
                y_zero_predicted = np.zeros(len(X_batch_and_covariates_no_ref_batch[0]))
                y_zero_predicted_nobatch = np.zeros(len(X_batch_and_covariates_no_ref_batch[0]))
            elif self.dict_logit_with_batch[feature][1] == 'all_equal_to_zero':
                Xt[:, feature] = 0
                continue
            else:
                Y_predicted = self.dict_logit_with_batch[feature][0].predict_proba(X_batch_and_covariates_no_ref_batch)
                y_zero_predicted = Y_predicted[:, 0]
                logit_model_nobatch = sklearn.linear_model.LogisticRegression(self.penalty,
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
                logit_model_nobatch.coef_ = self.dict_logit_with_batch[feature][0].coef_.copy()
                logit_model_nobatch.coef_[0, self.batch_columns] = 0
                Y_predicted_nobatch = logit_model_nobatch.predict_proba(X_batch_and_covariates_no_ref_batch)
                y_zero_predicted_nobatch = Y_predicted_nobatch[:, 0]
            predictions = dict()
            predictions_nobatch = dict()
            counter_quantile = 0
            for quantile in self.quantiles:
                predictions[quantile] = self.dict_quantile_with_batch[feature][counter_quantile].predict(
                    X_batch_and_covariates_no_ref_batch)
                if feature in jittering_columns_indexes:
                    predictions[quantile] = np.floor(predictions[quantile])
                quantile_model_nobatch = sklearn.linear_model.QuantileRegressor(quantile=quantile,
                                                                                alpha=alphas_list[feature],
                                                                                fit_intercept=self.fit_intercept_quantile,
                                                                                solver=self.solver_quantile,
                                                                                solver_options=self.solver_options
                                                                                )
                quantile_model_nobatch.coef_ = self.dict_quantile_with_batch[feature][counter_quantile].coef_.copy()
                quantile_model_nobatch.coef_[self.batch_columns] = 0
                predictions_nobatch[quantile] = quantile_model_nobatch.predict(X_batch_and_covariates_no_ref_batch)
                if feature in jittering_columns_indexes:
                    predictions_nobatch[quantile] = np.floor(predictions_nobatch[quantile])
                counter_quantile += 1
            for sample in range(len(y_initial)):
                predictions_correct = dict()
                predictions_correct_nobatch = dict()
                quantile_correct_list = []
                quantile_correct_nobatch_list = []
                for quantile in self.quantiles:
                    quantile_correct = 1 - (1 - quantile) * (1 - y_zero_predicted[sample])
                    quantile_correct_list.append(quantile_correct)
                    predictions_correct[quantile_correct] = predictions[quantile][sample]
                    quantile_correct_nobatch = 1 - (1 - quantile) * (1 - y_zero_predicted_nobatch[sample])
                    quantile_correct_nobatch_list.append(quantile_correct_nobatch)
                    predictions_correct_nobatch[quantile_correct_nobatch] = predictions_nobatch[quantile][sample]
                y_initial_sample = y_initial[sample]
                y_initial_sample_quantile = y_zero_predicted[sample]
                for quantile in quantile_correct_list:
                    if predictions_correct[quantile] <= y_initial_sample:
                        y_initial_sample_quantile = quantile
                y_nobatch = 0
                for quantile in quantile_correct_nobatch_list:
                    if quantile <= y_initial_sample_quantile:
                        y_nobatch = predictions_correct_nobatch[quantile]
                Xt[sample, feature] = y_nobatch
        return Xt
