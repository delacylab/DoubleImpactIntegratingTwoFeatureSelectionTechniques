# Overview
_Double impact_ is an all-relevant feature selection pipeline that integrates two existing methods: cross-validated LASSO and Boruta. The former, also commonly known as [LASSOCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html), is a popular feature selection method based on linear/logistic regression with L1-regularization. [Boruta](https://content.iospress.com/articles/fundamenta-informaticae/fi101-4-02), on the other hand, relies on an ensemble-based method of tree models (i.e., random forest method) and multiple hypothesis testing to perform feature selection. 

This integration emphasizes how the two mentioned methods are complementary. LASSOCV captures the _linearly relevant_ feature subset through L1-regularization. However, any feature relevant in a non-linear manner might not be captured by LASSOCV. Thus, a type-II error (i.e., false negative) can be committed when LASSOCV rejects the non-linearly relevant feature as irrelevant. Analogously, the random forest architecture omits the linearity of the data so a linearly relevant feature can be deemed irrelevant according to the majority votes from its decision trees and the statistical tests performed internally. Hence, it is desirable to have a general pipeline that minimizes the type-II error committed on either side. 

# Modification to BorutaPy
In practice, users may want to analyze _more than one_ feature subset because of the possible errors made by the feature selection method. For instance, by decreasing the significance level in a statistical test (e.g., from $\alpha=0.05$ to $0.01$), Boruta can accept strictly fewer features. This motivates the design of a feature selection pipeline that allows users to fine-tune different hyperparameters to get different selected feature subsets for later modeling purposes. 

In the original Boruta R-implementation, acceptance/rejection of a feature depends on its importance score compared to the _maximum_ importance score of the shadow features (i.e., features created by shuffling the samples of the real features). The [BorutaPy Python-implementation](https://github.com/scikit-learn-contrib/boruta_py) relaxes this condition by allowing users to compare each real feature to a specific percentile (_perc_) of the shadow features' importance scores. They also introduced a Boolean flag (_two_step_) to decide whether a two-step p-values correction process is used in the multiple hypothesis testing (compared to the one-step Bonferroni correction in the original work). 

Noticeably, setting _perc_, _alpha_, or _two_step_ to a different value can yield drastically different feature subsets. While BorutaPy only allows one value input for each mentioned hyperparameter, our pipeline permits a list of values as input. For example, our pipeline allows _perc=[100, 90]_ such that only one run is required for different _perc_ values, and results in a shorter execution time compared to BorutaPy. Nevertheless, this optimization method comes with a cost. In each iteration, different hyperparameter configurations correspond to different criteria to reject a feature. To account for their differences, the user can choose a decision rule _hp_rule_ to decide when a feature will be removed from the upcoming iterations.

[Unanimous rule] (_hp_rule_=0) A feature is rejected if _all_ hyperparameter configurations agree to reject the feature.

[Majority rule] (_hp_rule_=1) A feature is rejected if _at least half_ of hyperparameter configurations agree to reject the feature.

[Minority rule] (_hp_rule_=2) A feature is rejected if _at least one_ hyperparameter configuration rejects the feature.

A higher value of _hp_rule_ is associated with a harsher selection criterion (at the risk of type-II error). The default value is set as _hp_rule_=0 to control for type-II errors. 

Additionally, tentative features (i.e., neither accepted nor rejected) are considered in the original Boruta method. When the median importance score of a tentative feature over all iterations is greater than the overall median of all features, Boruta deems it _weakly accepted_ by a _fixing procedure_. One crucial merit of this procedure is to control for type-II errors. However, our pipeline does not consider weakly accepted features because we control type-II errors by offering a wider choice of hyperparameter configurations. Users are suggested to use BorutaPy if they are interested in the weakly accepted feature subset. We will consider embedding their fixing procedure in a future release.  

# How to use the pipeline
Users are suggested to use the original codes directly from [LASSOCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html) or [BorutaPy Python-implementation](https://github.com/scikit-learn-contrib/boruta_py) if they are interested in only the result of LASSOCV or BorutaPy but not both. Also, if they are interested in only running Boruta with a single hyperparameter configuration. 

The script _Boruta_Revised.py is largely based on the BorutaPy implementation with the mentioned modifications. By providing a sklearn estimator (e.g., RandomForestRegressor or RandomForestClassifer) and fitting the data (by ._fit), the class _BorutaClass_ has an attribute _.decision_dict_ to encode if a feature is accepted (1), rejected (-1), or tentative (0) in each hyperparameter configurations. 

The script _Double_Impact.py integrates _Boruta_Revised.py with LASSOCV from sklearn as a function. While all runtime parameters have their default values specified as those in sklearn or BorutaPy, the function returns a Pandas.DataFrame that summarizes the selection results and also the _BorutaClass_ object. Below is an example of the summary data frame.

|Feature|LASSOCV_coefficient|LASSOCV_beta_L_0|Boruta_perc_100_alpha_0.05_two_step_True|Boruta_perc_90_alpha_0.05_two_step_True|
| --- | --- | --- | --- | --- |
| X1 | 0 | 0 | 0 | 1 |
| X2 | 0.5 | 1 | 1 | 1 |

Results of feature selection are encoded by binary values starting in the third column (0: not selected, 1: selected).  

The bottom of each script provided a simple execution example.
