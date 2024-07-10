########################################################################################################################
# Last update: 2024-07-07 15:40 MDT by Wayne Lam
########################################################################################################################

# Disclaimer:
# A large portion of the code is borrowed from BorutaPy (by Daniel Homola <dani.homola@gmail.com>). BorutaPy is a Python
# implementation of the R implementation of the Boruta method originated from Miron B. Kursa, Aleksander Jankowski, and
# Witold R. Rudnicki.
# (1) For the original method, see Kursa, Jankowski, and Rudnicki. Boruta - A System for Feature Selection.
# https://content.iospress.com/articles/fundamenta-informaticae/fi101-4-02
# (2) For the R implementation of Boruta, see Kursa & Rudnickl. Feature Selection with the Boruta Package.
# https://www.jstatsoft.org/article/view/v036i11
# (3) For the BorutaPy Python implementation, see https://github.com/scikit-learn-contrib/boruta_py.

# Overview:
# Compared to the BorutaPy package, the main modification made in our script is how different values of hyperparameters
# can be executed in a single call without redoing the modeling procedure multiple times. For example, as suggested in
# the BorutaPy documentation, using the maximum importance scores from the shadow features "could be overly harsh". So,
# they introduced percentile (perc) to control for the strictness of comparison. However, it is hard to know in advance
# what a 'good' value of perc should be chosen. The goodness, practically speaking, depends on how the performance
# statistics obtained by a certain modeling method with the selected feature subset. Thus, it is desirable to obtain
# multiple feature subsets for later comparison.

# Remark 1 - Hyperparameter Tuning:
# Users can fine-tune 3 hyperparameters: perc, alpha, and two_step. Each can be specified as a list of values instead
# of a single value in the BorutaPy implementation:
# (a) perc controls the percentile of shadow features' importances scores that each real feature is compared to in
# each iteration (greater perc is harsher),
# (b) alpha controls the level of significance to accept/reject a statistical test (smaller alpha is harsher), and
# (c) two_step controls whether two-step p-value correction is used instead of the original one-step Bonferroni
# correction (setting False is harsher).
# On the other hand, notice that, in the original implementation, once a feature is deemed irrelevant (rejected) in a
# given iteration, it will not be considered in later iterations in the random forest modeling process. However, when
# considering multiple hyperparameter configurations, they can disagree on whether a feature should be deemed irrelevant
# or not. To solve this problem, we allow users to pick from one of the three rules (in 'hp_rule').
# [Rule 1, unanimous rule] A feature is truly rejected and will not be used for fitting in later iterations if ALL
# hyperparameter configurations reject the feature.
# [Rule 2, majority rule]  A feature is truly rejected and will not be used for fitting in later iterations if AT LEAST
# HALF of the hyperparameter configurations reject the feature.
# [Rule 3, minority rule]  A feature is truly rejected and will not be used for fitting in later iterations if AT LEAST
# ONE hyperparameter configuration rejects the feature.
# Obviously, rule 1 is the strongest rule to avoid Type-2 error whereas rule 3 is the weakest.

# Remark 2 - Fixed Tentative Features Not Included in the Final Output
# While both the original R implementation and the BorutaPy implementation consider tentative features (i.e., features
# neither accepted nor rejected as relevant) with a fixing procedure to determine if a tentative feature should be
# weakly accepted (when its median importance score over iterations is greater than the overall median). One crucial
# merit of doing so is how users can minimize type-II error by considering not just the accepted feature subset,
# but also the weakly accepted subset. However, we choose to minimize type-II error by offering a wider choice of
# hyperparameter configurations. For users who are interested in the weakly accepted feature subset of a given
# hyperparameter configuration, they are suggested to use the original BorutaPy implementation.

########################################################################################################################
# Import packages
########################################################################################################################
from __future__ import print_function, division
import numpy as np
import pandas as pd
import scipy as sp
from itertools import product
from sklearn.utils import check_random_state, check_X_y
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import _is_fitted

########################################################################################################################
# Define Boruta model class
########################################################################################################################


class BorutaClass(BaseEstimator, TransformerMixin):
    """
    Runtime parameters
    ------------------
    (1) estimator: (object)
        [Identical to BorutaPy]
        A supervised learning estimator, with a 'fit' method that returns the feature_importances_ attribute. Important
        features must correspond to high absolute values in the feature_importances_.
    (2) n_estimators: (int or str, default = 1000)
        [Identical to BorutaPy]
        If int, set the number of estimators in the chosen ensemble method.
        If 'auto', the number of estimators is determined automatically based on the size of the dataset.
    (3) perc: (int or list of int, default = 100)
        [Different from BorutaPy]
        Each perc is an integer controlling how a real feature is compared to the set of shadow features. When perc is
        set as k for 1 <= k <= 100, the importance score of the real feature is compared to the shadow feature with
        the k-th percentile importance score. Smaller perc values correspond to a higher risk of type-I error. Unlike
        the original implementation in BorutaPy that allows only a single input integer, we allow the input as a list
        of integers in case the user is interested to tune this hyperparameter to obtain different feature subsets
        (without running the models repeatedly for different configurations).
    (4) alpha: (float or list of floats, default = 0.05)
        [Different from BorutaPy]
        Level of significance in hypothesis testing. Similar to perc in (3), we allow the input as a list of floats in
        case the user is interested to tune this hyperparameter.
    (5) two_step: (Boolean or list of Boolean, default = True)
        [Different from BorutaPy]
        0: Boruta with Bonferroni correction for p-values
        1: Boruta without Bonferroni correction for p-values
        2: Boruta with and without (separately) Bonferroni correction for p-values. Similar to (3) and (4), this setting
        allows users to compare how Bonferroni correction impacts the feature selection process.
    (6) max_iter: (int, default=100)
        [Identical to BorutaPy]
        The number of maximum iterations to perform.
    (7) hp_rule: (int in [0, 1, 2], default=0)
        [Different from BorutaPy]
        [Unanimous rule] If 0, a feature is truly rejected (and will not be used for fitting in later iterations)
        when ALL hyperparameter configurations reject the feature.
        [Majority rule] If 1, a feature is truly rejected (and will not be used for fitting in later iterations)
        when AT LEAST HALF of the hyperparameter configurations reject the feature.
        [Minority rule] If 2, a feature is truly rejected (and will not be used for fitting in later iterations)
        when AT LEAST ONE of the hyperparameter configurations reject the feature.
    (8) random_state: (int, RandomState instance or None; default=None)
        [Identical to BorutaPy]
        If int or RandomState instance, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used by np.random.
    (9) verbose: (int in [0, 1], default=0)
        [Different from BorutaPy]
        If 0, no output will be printed to the console.
        If 1, print the indices of the selected feature subset for each hyperparameter configuration.

    Attributes
    ----------
    (a) _get_shuffle(input_array)
        Shuffle the input_array
    (b) _get_shadow(input_array)
        Obtain the shadow copy of the features in the input_array
    (c) _get_importance(X, y)
        Fit a random forest model (i.e., estimator) with (X, y) and return the importance scores of each feature in X.
    (d) _get_model()
        Obtain the (fitted) random forest model in the last iteration.
    (e) _get_tree_num(n_feat)
        Compute the optimal number of trees (i.e., n_estimators) relative to a given number of features.
    (f)  _get_rejection()
        Return two lists of indices: first for the rejected features and second for the features not rejected.
    (g) _fdrcorrection(pvals)
        Perform FDR correction for a given list of p-values pvals.
    (h) _do_tests(n_iter)
        Perform statistical tests for feature importance after a given number of iterations n_iter.
    (i) _fit(X, y) and fit(X, y)
        Fit the Boruta model with (X, y).
    (j) _get_accepted(perc, alpha, two_step)
        Return the list of indices of the accepted features relative to the specified hyperparameter setting.
    (k) _transform(X, perc alpha, two_step)
        Transform the full feature set into its feature subset relative to the specified hyperparameter setting.
    (l) fit_transform(X, y, perc, alpha, two_step)
        Return the transformed feature subset relative to the specified hyperparameter setting after model fitting.
    (m) importance_history
        A 2-d array recording the importance scores of the real features in all iterations.
    (n) hit_dict
        A dictionary with keys as the hyperparameter configurations and values as the number of hits for each feature
        over all iterations.
    (o) decision_dict
        A dictionary with keys as the hyperparameter configurations and values as the decision made to each feature
        1: accepted, -1: rejected, 0: tentative (neither accepted nor rejected)
    """
    def __init__(self, estimator, n_estimators=1000, perc=100, alpha=0.05, two_step=True,
                 max_iter=100, hp_rule=0, random_state=None, verbose=0):
        assert (n_estimators == 'auto') or (type(n_estimators) is int and n_estimators > 0), \
            f"Number greater than 0 expected, current input: {n_estimators}"
        assert (type(perc) is int and 1 <= perc <= 100) or all([(type(p) is int) and (1 <= p <= 100) for p in perc]), \
            f"Number (or list of numbers with each) within [1, 100] expected; current input: {perc}"
        assert (type(alpha) is float and 0 < alpha < 1) or all([(type(a) is float) and (0 < a < 1) for a in alpha]), \
            f"Float (or list of floats with each) within (0, 1) expected; current input: {alpha}"
        assert type(two_step) is bool or all([(type(t) is bool) for t in two_step]), \
            f"Boolean (or list of Boolean with each) in [True, False] expected; current input: {two_step}"
        assert type(max_iter) is int and max_iter > 0, f"Integer greater than 0 expected; current input: {max_iter}"
        assert hp_rule in [0, 1, 2], f"Integer in [0, 1, 2] expected; current input: {hp_rule}"
        assert verbose in [0, 1], f"Integer in [0, 1] expected; current input: {verbose}"

        self.estimator = estimator
        self.n_estimators = n_estimators
        self.perc = perc
        self.alpha = alpha
        self.two_step = two_step
        self.max_iter = max_iter
        self.hp_rule = hp_rule
        self.random_state = check_random_state(random_state)
        self.verbose = verbose

        # Specify all the hyperparameter configurations
        self.perc_list = perc if type(perc) is not int else [perc]
        self.alpha_list = alpha if type(alpha) is not float else [alpha]
        self.two_step_list = two_step if type(two_step) is not bool else [two_step]
        self.hit_dict = {}
        self.decision_dict = {}
        for hp_config in product(self.perc_list, self.alpha_list, self.two_step_list):  # Hyperparameter configurations
            self.hit_dict[hp_config], self.decision_dict[hp_config] = None, None
        self.importance_history = None

    def _get_shuffle(self, input_array):
        self.random_state.shuffle(input_array)
        return input_array

    def _get_shadow(self, input_array):
        X_shadow = np.copy(input_array)
        while X_shadow.shape[1] < 5:
            X_shadow = np.hstack((X_shadow, X_shadow))
        X_shadow = np.apply_along_axis(self._get_shuffle, 0, X_shadow)
        return X_shadow

    def _get_importance(self, X, y):
        try:
            self.estimator.fit(X, y)
        except Exception as e:
            raise ValueError("Please check your X and y. The provided estimator cannot be fitted to your data."
                             "\n" + str(e))
        try:
            imp = self.estimator.feature_importances_
        except Exception:
            raise ValueError("Only methods with feature_importance_ attribute are currently supported.")
        return imp

    def _get_model(self):
        """
        Return the estimator (usually after model fitting) in case the user need it. [Different from BorutaPy]
        :return: The estimator of Boruta
        """
        return self.estimator

    def _get_tree_num(self, n_feat):
        depth = self.estimator.get_params()['max_depth']
        if depth is None:
            depth = 10
        f_repr = 100            # how many times a feature should be considered on average
        multi = ((n_feat * 2) / (np.sqrt(n_feat * 2) * depth))
        # n_feat * 2 because the training matrix is extended with n shadow features
        n_estimators = int(multi * f_repr)
        return n_estimators

    def _get_rejection(self):
        first_dec_reg = list(self.decision_dict.values())[0]
        assert first_dec_reg is not None
        full_features = set(range(len(first_dec_reg)))
        if self.hp_rule == 0:       # unanimous rule
            not_rejected = set()
            for dec_reg in self.decision_dict.values():
                not_rejected |= set(np.where(dec_reg >= 0)[0])
            rejected = full_features.difference(not_rejected)
        elif self.hp_rule == 1:     # majority rule
            n_hp_config = len(self.decision_dict.keys())
            rejected_dict = {feat: 0 for feat in full_features}
            for dec_reg in self.decision_dict.values():
                rejected_cur = np.where(dec_reg == -1)[0]
                for feat in rejected_cur:
                    rejected_dict[feat] += 1
            rejected = set(feat for feat, count in rejected_dict.items() if count / n_hp_config >= 0.5)
            not_rejected = full_features.difference(rejected)
        else:
            rejected = set()
            for dec_reg in self.decision_dict.values():
                rejected |= set(np.where(dec_reg == -1)[0])
            not_rejected = full_features.difference(rejected)
        return sorted(rejected), sorted(not_rejected)

    def _fdrcorrection(self, pvals):     # [Identical to BorutaPy]
        pvals = np.asarray(pvals)
        pvals_sortind = np.argsort(pvals)
        pvals_sorted = np.take(pvals, pvals_sortind)
        nobs = len(pvals_sorted)
        ecdffactor = np.arange(1, nobs+1) / float(nobs)
        pvals_corrected_raw = pvals_sorted / ecdffactor
        pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
        pvals_corrected[pvals_corrected > 1] = 1
        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[pvals_sortind] = pvals_corrected   # reorder p-values to original order of pvals
        return pvals_corrected_

    def _do_tests(self, n_iter):
        for hp_config in self.decision_dict.keys():
            perc, alpha, two_step = hp_config[0], hp_config[1], hp_config[2]
            dec_reg = self.decision_dict[hp_config]
            hit_reg = self.hit_dict[hp_config]
            active_features = np.where(dec_reg >= 0)[0]
            hits = hit_reg[active_features]

            # get uncorrected p-values based on hit_reg
            to_accept_ps = sp.stats.binom.sf(hits-1, n_iter, .5).flatten()
            to_reject_ps = sp.stats.binom.cdf(hits, n_iter, .5).flatten()

            if two_step:
                to_accept = self._fdrcorrection(to_accept_ps)
                to_reject = self._fdrcorrection(to_reject_ps)
                to_accept2 = to_accept_ps <= alpha / float(n_iter)
                to_reject2 = to_reject_ps <= alpha / float(n_iter)
                to_accept *= to_accept2
                to_reject *= to_reject2
            else:
                to_accept = to_accept_ps <= alpha / float(len(dec_reg))
                to_reject = to_reject_ps <= alpha / float(len(dec_reg))

            to_accept = np.where((dec_reg[active_features] == 0) * to_accept)[0]
            to_reject = np.where((dec_reg[active_features] == 0) * to_reject)[0]

            # update dec_reg
            dec_reg[active_features[to_accept]] = 1
            dec_reg[active_features[to_reject]] = -1
            self.decision_dict[hp_config] = dec_reg

    def _fit(self, X, y):
        """Fit the Boruta feature selection with the provided estimator. [Roughly identical to _fit in BorutaPy]
        :param X: array-like. The training input samples with shape = [n_samples, n_features].
        :param y: array-like. The target values of the training set with shape = [n_samples]
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        n_sample, n_feat = X.shape
        _iter = 1
        for hp_config in product(self.perc_list, self.alpha_list, self.two_step_list):
            if self.hit_dict[hp_config] is None:
                self.hit_dict[hp_config] = np.zeros(n_feat, dtype=np.int32)
            if self.decision_dict[hp_config] is None:
                self.decision_dict[hp_config] = np.zeros(n_feat, dtype=np.int32)
        if self.importance_history is None:
            self.importance_history = np.zeros(n_feat, dtype=np.int32)

        # set n_estimators if not already specified
        if self.n_estimators != 'auto':
            self.estimator.set_params(n_estimators=self.n_estimators)

        # main feature selection loop
        while _iter <= self.max_iter and any([np.any(v == 0) for v in self.decision_dict.values()]):
            # the second condition above ensures that the loop will continue unless all hp_config have no features
            # left un-rejected or un-accepted.
            rejected, unrejected = self._get_rejection()
            n_unrejected = len(unrejected)
            if self.n_estimators == 'auto':
                # find optimal number of trees and depth
                n_tree = self._get_tree_num(n_unrejected)
                self.estimator.set_params(n_estimators=n_tree)
            else:
                n_tree = self.n_estimators

            # Obtain shadow features
            X_unrejected = X[:, unrejected]
            X_shadow = self._get_shadow(X_unrejected)
            if self.verbose:
                print(f"Iteration {_iter}: Fitting a Random Forest model (of {n_tree} trees) with {n_unrejected} real "
                      f"features and {X_shadow.shape[1]} shadow features.", flush=True)

            # Fit model and get importances
            importance = self._get_importance(np.hstack((X_unrejected, X_shadow)), y)
            importance_shadow = importance[n_unrejected:]
            importance_real = np.full(X.shape[1], np.nan)
            importance_real[unrejected] = importance[:n_unrejected]
            self.importance_history = np.vstack((self.importance_history, importance_real))
            importance_real_no_nan = importance_real.copy()
            importance_real_no_nan[np.isnan(importance_real_no_nan)] = 0

            # Register hits according to different perc values
            for perc in self.perc_list:
                shadow_importance_target = np.percentile(importance_shadow, perc)
                hits = np.where(importance_real_no_nan > shadow_importance_target)[0]
                for k in self.hit_dict.keys():
                    if k[0] == perc:
                        self.hit_dict[k][hits] += 1

            self._do_tests(_iter)
            _iter += 1
            if self.verbose:
                for config_idx, hp_config in enumerate(self.decision_dict.keys(), 1):
                    perc, alpha, two_step = hp_config[0], hp_config[1], hp_config[2]
                    dec_reg = self.decision_dict[hp_config]
                    print(f"Configuration {config_idx}; (perc={perc}, alpha={alpha}, two_step={two_step}). "
                          f"Accepted indices: {np.where(dec_reg == 1)[0]}", flush=True)
                print("\n", flush=True)

        if self.verbose:
            print(f"Boruta finished running. Use the attribute .decision_dict to obtain a summary of each feature if "
                  f"interested (1: accepted, -1: rejected, 0: tentative (neither accepted nor rejected).\n")

    def _get_accepted(self, perc, alpha, two_step):
        assert perc in self.perc_list
        assert alpha in self.alpha_list
        assert two_step in self.two_step_list
        if not _is_fitted(self.estimator):
            return []
        deg_reg = self.decision_dict[(perc, alpha, two_step)]
        return np.where(deg_reg == 1)[0]

    def _transform(self, X, perc, alpha, two_step):
        if not _is_fitted(self.estimator):
            raise ValueError('You need to call the fit(X, y) method first.')
        support_ = self._get_accepted(perc, alpha, two_step)
        return X[:, support_]

    def fit(self, X, y):
        """Fit the Boruta feature selection with the provided estimator. [Identical to BorutaPy]
        :param X: array-like. The training input samples with shape = [n_samples, n_features].
        :param y: array-like. The target values of the training set with shape = [n_samples]
        """
        return self._fit(X, y)

    def fit_transform(self, X, y, perc, alpha, two_step):
        self._fit(X, y)
        return self._transform(X, perc, alpha, two_step)


########################################################################################################################

if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    X, y = make_classification(n_samples=100, n_features=10, n_informative=2, random_state=42, shuffle=False)
    M = RandomForestClassifier(class_weight='balanced', max_depth=5, bootstrap=False,
                               random_state=42, n_jobs=-1)
    B = BorutaClass(estimator=M, n_estimators='auto', max_iter=30,
                    perc=[100, 80, 60], alpha=0.05, two_step=[True, False],
                    hp_rule=1, random_state=42, verbose=1)
    B.fit(X, y)
    for config_idx, (k, v) in enumerate(B.decision_dict.items(), 1):
        perc, alpha, two_step = k[0], k[1], k[2]
        print(f"Configuration {config_idx}; (perc={perc}, alpha={alpha}, two_step={two_step}). "
              f"Decision: {v}", flush=True)
