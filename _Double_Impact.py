########################################################################################################################
# Last update: 2024-07-07 15:40 MDT by Wayne Lam
########################################################################################################################
import numpy as np
import pandas as pd
from _Boruta_Revised import BorutaClass
from sklearn.linear_model import LassoCV

########################################################################################################################


def featureSelect(X, y, estimator_B, beta_L=0, eps_L=1e-3, n_alphas_L=100, alphas_L=None, fit_intercept_L=True,
                  precompute_L='auto', max_iter_L=1000, tol_L=1e-4, copy_X_L=True, cv_L=5, verbose_L=False,
                  n_jobs_L=None, positive_L=False, random_state_L=None, selection_L='cyclic',
                  n_estimators_B=1000, perc_B=100, alpha_B=0.05, two_step_B=True,
                  max_iter_B=100, hp_rule_B=0, random_state_B=None, verbose_B=0, verbose=0):
    """
    This preprocessing pipeline is an all-relevant feature selection method that integrates two existing methods:
    LASSOCV (from sklearn packaage) and BorutaPy (from boruta package). In particular, the latter has been modified to
    allow running different hyperparameter configurations at the same time. This integration emphasizes how the two
    existing methods are complementary. LASSOCV captures the linearly relevant feature subset through L1-regularization,
    and can commit a type-II error when a feature is only non-linearly relevant. Analogously, Boruta, as an
    ensemble-based method of decision trees, omits linearity and can commit a type-II error when a feature is linearly
    relevant but not deemed relevant by the majority votes from its decision trees. So, the integration aims to capture
    both linearly and non-linearly relevant features.
    Users are suggested to use the original codes if they are interested in only the result of LASSOCV or Boruta but not
    both.
    Runtime parameters
    ------------------
    (1) X: array-like
    The training feature dataset with shape = [n_samples, n_features]
    (2) y: array-like
    The training target with shape = [n_samples]
    (3) estimator_B: (object)
    A supervised learning estimator, with a 'fit' method that returns the feature_importances_ attribute. Important
    features must correspond to high absolute values in the feature_importances_. We expect this to be a
    sklearn.ensemble Random Forest estimator object (RandomForestClassifier for classification or RandomForestRegressor
    for regression).
    (4) beta_L: (non-negative float, default: 0)
    A feature is deemed linearly relevant by LASSOCV if the absolute value of its regression coefficient is greater than
    the threshold.
    (5) eps_L: (non-negative float, default: 1e-3)
    eps in sklearn.linear_model.LassoCV that controls the length of the regularization path.
    (6) n_alphas_L: (positive integer, default: 100)
    n_alphas in sklearn.linear_model.LassoCV. Number of alphas along the regularization path.
    (7) alphas_L: (list of positive floats or None, default: None)
    alphas in sklearn.linear_model.LassoCV. List of alphas to compute the models. Computed automatically if None.
    (8) fit_intercept_L: (Boolean, default: True)
    fit_intercept in sklearn.linear_model.LassoCV. Whether to calculate the intercept for the linear model.
    (9) precompute_L: (Boolean, 'auto' or array-like, default: 'auto')
    precompute in sklearn.linear_model.LassoCV. Whether to use a precomputed Gram matrix to speed up calculations.
    (10) max_iter_L: (positive integer, default: 1000)
    max_iter in sklearn.linear_model.LassoCV. Maximum number of iterations to fit the linear model.
    (11) tol_L: (non-negative float, default: 1e-4)
    tol in sklearn.linear_model.LassoCV. The tolerance for the optimization.
    (12) copy_X_L: (Boolean, default: True)
    copy_X in sklearn.linear_model.LassoCV. X will be copied if True, overwritten otherwise.
    (13) cv_L: (positive integer, a cross-validation generator, an iterable, or None, default: None)
    cv in sklearn.linear_model.LassoCV. An integer, cross-validation generator or iterable to perform cross-validation.
    5-fold cross-validation if None.
    (14) verbose_L: (Boolean, default: False)
    verbose in sklearn.linear_model.LassoCV. Amount of verbosity of the LASSOCV subroutine.
    (15) n_jobs_L: (positive integer, -1, or None default = None)
    n_jobs in sklearn.linear_model.LassoCV. Number of CPUs to use during cross validation. One CPU if None and all CPUs
    if -1.
    (16) positive_L: (boolean, default: False)
    positive in sklearn.linear_model.LassoCV. Only positive regression coefficients if True.
    (17) random_state_L: (random_state, default: None)
    random_state in sklearn.linear_model.LassoCV. Seed of the pseudo random number generator when 'selection_L'='random.
    (18) selection_L: ('random' or 'cyclic', default: 'cyclic)
    selection in sklearn.linear_model.LassoCV. Random coefficient updates in every iteration if 'random', and looping
    over features if 'cyclic'.
    (19) n_estimators_B: (int or str, default = 1000)
    If int, set the number of estimators in the chosen ensemble method.
    If 'auto', the number of estimators is determined automatically based on the size of the dataset.
    (20) perc_B: perc: (int or list of int, default = 100)
    Each perc is an integer controlling how a real feature is compared to the set of shadow features. When perc is
    set as k for 1 <= k <= 100, the importance score of the real feature is compared to the shadow feature with
    the k-th percentile importance score. Smaller perc values correspond to a higher risk of type-I error. Unlike
    the original implementation in BorutaPy that allows only a single input integer, we allow the input as a list
    of integers in case the user is interested to tune this hyperparameter to obtain different feature subsets
    (without running the models repeatedly for different configurations).
    (21) alpha_B: (float or list of floats, default = 0.05)
    Level of significance in hypothesis testing. Similar to perc in (3), we allow the input as a list of floats in
    case the user is interested to tune this hyperparameter.
    (22) two_step: (Boolean or list of Boolean, default = True)
    0: Boruta with Bonferroni correction for p-values
    1: Boruta without Bonferroni correction for p-values
    2: Boruta with and without (separately) Bonferroni correction for p-values. Similar to (3) and (4), this setting
    allows users to compare how Bonferroni correction impacts the feature selection process.
    (23) max_iter_B: (int, default=100)
    The number of maximum iterations to perform.
    (24) hp_rule_B: (int in [0, 1, 2], default=0)
    [Unanimous rule] If 0, a feature is truly rejected (and will not be used for fitting in later iterations)
    when ALL hyperparameter configurations reject the feature.
    [Majority rule] If 1, a feature is truly rejected (and will not be used for fitting in later iterations)
    when AT LEAST HALF of the hyperparameter configurations reject the feature.
    [Minority rule] If 2, a feature is truly rejected (and will not be used for fitting in later iterations)
    when AT LEAST ONE of the hyperparameter configurations reject the feature.
    (25) random_state_B: (int, RandomState instance or None; default=None)
    If int or RandomState instance, random_state is the seed used by the random number generator.
    If None, the random number generator is the RandomState instance used by np.random.
    (26) verbose_B:  (int in [0, 1], default=0)
    If 0, no output will be printed to the console.
    If 1, print the indices of the selected feature subset for each hyperparameter configuration.
    (27) verbose: (int in [0, 1], default=0)
    If 0, no output of the overall pipeline will be printed to the console.
    If 1, print the process of the overall pipeline to the console.
    :return:
    df_output: a Pandas DataFrame summarizing the feature selection result
    B_selector: the fitted Boruta model. See _Boruta_Revised.py for its attributes.
    """
    ####################################################################################################################
    # Prepare output summary file
    ####################################################################################################################
    if verbose == 1:
        print(f"Dimension of the dataset: {X.shape}\n", flush=True)
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
        X_values = X.values
    else:
        feature_names = list(range(X.shape[1]))
        X_values = X
    df_output = pd.DataFrame(feature_names, columns=['Feature'])

    ####################################################################################################################
    # Subroutine 1: Obtain linearly relevant features from LASSOCV
    ####################################################################################################################
    lasso_selector = LassoCV(eps=eps_L, n_alphas=n_alphas_L, alphas=alphas_L, fit_intercept=fit_intercept_L,
                             precompute=precompute_L, max_iter=max_iter_L, tol=tol_L, copy_X=copy_X_L,
                             cv=cv_L, verbose=verbose_L, n_jobs=n_jobs_L, positive=positive_L,
                             random_state=random_state_L, selection=selection_L)
    if verbose:
        print(f"Fitting LASSOCV model...", flush=True)
    lasso_selector.fit(X_values, y)

    df_output['LASSOCV_coefficient'] = lasso_selector.coef_
    df_output[f'LASSOCV_beta_L_{beta_L}'] = (df_output['LASSOCV_coefficient'].abs() > beta_L).astype(int)
    n_lasso_feat = df_output[df_output[f'LASSOCV_beta_L_{beta_L}'] == 1].shape[0]
    if verbose:
        print(f"{n_lasso_feat} linearly relevant features are returned by LASSOCV.\n", flush=True)

    ####################################################################################################################
    # Subroutine 2: Obtain non-linearly relevant features from the modified Boruta module
    ####################################################################################################################
    B_selector = BorutaClass(estimator=estimator_B, n_estimators=n_estimators_B, perc=perc_B, alpha=alpha_B,
                             two_step=two_step_B, max_iter=max_iter_B, hp_rule=hp_rule_B, random_state=random_state_B,
                             verbose=verbose_B)
    if verbose:
        print(f"Fitting Boruta model...", flush=True)
    B_selector.fit(X_values, y)
    for hp_config in B_selector.decision_dict.keys():
        perc, alpha, two_step = hp_config[0], hp_config[1], hp_config[2]
        dec_reg = B_selector.decision_dict[hp_config]
        col_name = f'Boruta_perc_{perc}_alpha_{alpha}_two_step_{two_step}'
        df_output[col_name] = dec_reg
        df_output[col_name] = (df_output[col_name] == 1).astype(int)

    return df_output, B_selector


########################################################################################################################
# Example
########################################################################################################################
if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    pd.set_option('display.max_columns', 10)

    X_, y_ = make_classification(n_samples=100, n_features=12, n_informative=6, random_state=42)
    # Comment out the line below if you want the feature set to be a numpy array
    X_ = pd.DataFrame(X_, columns=[f'X_{i}' for i in range(X_.shape[1])])

    randomForestModel = RandomForestClassifier(random_state=42)
    df_output_, B_selector_ = featureSelect(X_, y_, estimator_B=randomForestModel,
                                            perc_B=[100, 80], alpha_B=[0.05, 0.01],
                                            max_iter_B=15, verbose_B=1, verbose=1)
    rejected_features = df_output_['Feature'].to_numpy()[B_selector_._get_rejection()[0]]
    print(f"Summary of preprocessing results:\n", df_output_)
    print(f"Feature truly rejected by Boruta: {rejected_features}")
