########################################################################################################################
# Last update: 2024-07-02 18:22 MDT
########################################################################################################################
import numpy as np
import pandas as pd
import warnings
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LassoCV

########################################################################################################################


def featureSelect(X, y, estimator_B, beta_L=0, eps_L=1e-3, n_alphas_L=100, alphas_L=None, fit_intercept_L=True,
                  precompute_L='auto', max_iter_L=1000, tol_L=1e-4, copy_X_L=True, cv_L=5, verbose_L=False,
                  n_jobs_L=None, positive_L=False, random_state_L=None, selection_L='cyclic',
                  n_estimators_B=1000, perc_B=100, alpha_B=0.05, two_step_B=True,
                  max_iter_B=100, verbose_B=0, verbose=1):
    """
    This preprocessing pipeline is an all-relevant feature selection method that integrates two existing methods:
    LASSOCV (from sklearn packaage) and BorutaPy (from boruta package). This integration emphasizes how the two existing
    methods are complementary. LASSOCV captures the linearly relevant feature subset through L1-regularization, and can
    commit a type-II error when a feature is only non-linearly relevant. Analogously, Boruta, as an ensemble-based
    method of decision trees, omits linearity and can commit a type-II error when a feature is linearly relevant but not
    deemed relevant by the majority votes from its decision trees. So, the integration aims to capture both linearly and
    non-linearly relevant features.
    Users are suggested to use the original codes if they are interested in only the result of LASSOCV or Boruta but not
    both.
    :param X: the training feature dataset
    (array-like, shape = [n_samples, n_features])
    :param y: the training target
    (array-like, shape = [n_samples])
    :param estimator_B: a sklearn.ensemble Random Forest estimator object (RandomForestClassifier for classification or
    RandomForestRegressor for regression).
    :param beta_L: a feature is deemed linearly relevant by LASSOCV if the absolute value of its regression coefficient
    is greater than the threshold.
    (positive float, default: 0)
    :param eps_L: eps in sklearn.linear_model.LassoCV that controls the length of the regularization path.
    (default: 1e-3)
    :param n_alphas_L: n_alphas in sklearn.linear_model.LassoCV. Number of alphas along the regularization path.
    (default: 100)
    :param alphas_L: alphas in sklearn.linear_model.LassoCV. List of alphas to compute the models. Computed
    automatically if None.
    (default: None)
    :param fit_intercept_L: fit_intercept in sklearn.linear_model.LassoCV. Whether to calculate the intercept for the
    linear model.
    (default: True)
    :param precompute_L: precompute in sklearn.linear_model.LassoCV. Whether to use a precomputed Gram matrix to
    speed up calculations.
    (default: 'auto')
    :param max_iter_L: max_iter in sklearn.linear_model.LassoCV. Maximum number of iterations to fit the linear model.
    (default: 1000)
    :param tol_L: tol in sklearn.linear_model.LassoCV. The tolerance for the optimization.
    (default: 1e-4)
    :param copy_X_L: copy_X in sklearn.linear_model.LassoCV. X will be copied if True, overwritten otherwise.
    (default: True)
    :param cv_L: cv in sklearn.linear_model.LassoCV. An integer, cross-validation generator or iterable to perform
    cross-validation. 5-fold cross-validation if None.
    (Default: None)
    :param verbose_L: verbose in sklearn.linear_model.LassoCV. Amount of verbosity of the LASSOCV subroutine.
    (Default: False)
    :param n_jobs_L: n_jobs in sklearn.linear_model.LassoCV. Number of CPUs to use during cross validation. One CPU if
    None and all CPUs if -1.
    (default: 1)
    :param positive_L: positive in sklearn.linear_model.LassoCV. Only positive regression coefficients if True.
    (default: False)
    :param random_state_L: random_state in sklearn.linear_model.LassoCV. Seed of the pseudo random number generator when
    'selection_L'=='random.
    (default: None)
    :param selection_L: selection in sklearn.linear_model.LassoCV. Random coefficient updates in every iteration if
    'random', and looping over features if 'cyclic'.
    (default: 'cyclic')
    :param n_estimators_B: n_estimators in boruta_py. Number of estimators for integers or determined automatically if
    'auto'.
    (default: 1000)
    :param perc_B: perc in boruta_py. Percentile of the importance scores from the shadow features that a real feature
    is compared with.
    (default: 100)
    :param alpha_B: alpha in boruta_py. Significance level to reject the null based on p-values.
    (Default: 0.05)
    :param two_step_B: two_step in boruta_py. Bonferroni correction of p-values if False.
    (Default: True)
    :param max_iter_B: max_iter in boruta_py. Maximum iterations to perform for the Boruta subroutine.
    (Default: 100)
    :param verbose_B: verbose in boruta_py. Amount of verbosity of the Boruta subroutine.
    (Default: 0)
    :param verbose: Amount of verbosity for the overall pipeline.
    (Default: 1)
    :return:
    """
    if verbose == 1:
        print(f"Dimension of the dataset: {X.shape}\n", flush=True)

    ####################################################################################################################
    # Prepare output summary file
    ####################################################################################################################
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
        X_values = X.values
    else:
        feature_names = list(range(X.shape[1]))
        X_values = X

    df_output = pd.DataFrame(feature_names, columns=['Feature'])
    lasso_feat_set, B_feat_set = set(), set()

    ####################################################################################################################
    # Subroutine 1: Obtain linearly important features from LASSOCV
    ####################################################################################################################
    lasso_selector = LassoCV(eps=eps_L, n_alphas=n_alphas_L, alphas=alphas_L, fit_intercept=fit_intercept_L,
                             precompute=precompute_L, max_iter=max_iter_L, tol=tol_L, copy_X=copy_X_L,
                             cv=cv_L, verbose=verbose_L, n_jobs=n_jobs_L, positive=positive_L,
                             random_state=random_state_L, selection=selection_L)
    if verbose:
        print(f"Fitting LASSO CV model...", flush=True)
    lasso_selector.fit(X_values, y)

    df_output['LASSOCV_coefficient'] = lasso_selector.coef_
    df_coef = pd.DataFrame(zip(feature_names, lasso_selector.coef_, ), columns=['features', 'coef'])
    df_coef['coef_abs'] = df_coef['coef'].abs()
    df_coef = df_coef[df_coef["coef_abs"] > beta_L]
    lasso_feat_set = set(df_coef['features'])
    if verbose:
        print(f"{len(lasso_feat_set)} linearly relevant features are returned by LASSOCV.", flush=True)

    ####################################################################################################################
    # Subroutine 2: Obtain non-linearly important features from Boruta
    ####################################################################################################################
    B_selector = BorutaPy(estimator=estimator_B, n_estimators=n_estimators_B, perc=perc_B, alpha=alpha_B,
                          two_step=two_step_B, max_iter=max_iter_B, verbose=verbose_B)
    if verbose:
        print(f"Fitting Boruta model...", flush=True)
    B_selector.fit(X_values, y)
    B_importances = B_selector._get_imp(X_values, y)
    B_support = B_selector.support_
    df_output['Boruta_importance'] = B_importances

    B_feat_set = set(feature_names[i] for i in range(len(feature_names)) if list(B_support)[i])
    if verbose:
        print(f"{len(B_feat_set)} non-linearly relevant features are returned by Boruta.", flush=True)

    ####################################################################################################################
    # Organized feature subsets
    ####################################################################################################################
    def selection(feat_name):
        if feat_name in lasso_feat_set:
            if feat_name in B_feat_set:
                return 'Both'
            else:
                return 'LASSOCV'
        elif feat_name in B_feat_set:
            return 'Boruta'
        else:
            return 'None'

    df_output['Selected_by'] = df_output['Feature'].apply(selection)
    union_LB = lasso_feat_set.union(B_feat_set)

    if verbose:
        intersect_LB_size = (len(lasso_feat_set) + len(B_feat_set)) - len(union_LB)
        linear_only_size = len(lasso_feat_set) - intersect_LB_size
        non_linear_only_size = len(B_feat_set) - intersect_LB_size
        print(f"Number of features that are either linearly or non-linearly relevant: {len(union_LB)}", flush=True)
        print(f"Number of features that are both linearly and non-linearly relevant: {intersect_LB_size}", flush=True)
        print(f"Number of features that are only linearly relevant: {linear_only_size}", flush=True)
        print(f"Number of features that are only non-linearly relevant: {non_linear_only_size}", flush=True)
        print(f"Returning results...\n")
    X_new = X[list(union_LB)]
    return X_new, df_output


########################################################################################################################
# Example
########################################################################################################################
if __name__ == '__main__':
    from sklearn.datasets import make_classification, make_regression
    pd.set_option('display.max_columns', None)

    X_, y_ = make_classification(n_samples=100, n_features=12, n_informative=6, random_state=42)
    # Comment out the line below if you want the feature set to be a numpy array
    X_ = pd.DataFrame(X_, columns=[f'X_{i}' for i in range(X_.shape[1])])

    randomForestModel = RandomForestClassifier(random_state=42)
    X_new_, summary = featureSelect(X_, y_, estimator_B=randomForestModel, max_iter_B=10, random_state_L=42)
    print(f"Preprocessed dataset:\n", X_new_, "\n")
    print(f"Summary of preprocessing results:\n", summary)
