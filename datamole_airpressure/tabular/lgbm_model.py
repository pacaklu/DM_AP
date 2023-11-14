"""Script for handling of lgbm model class."""

from typing import List, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split


class LGBMModel:
    """Class for lgbm model."""

    def __init__(self, params):
        """Load parameters of model."""
        self.params = params
        self.shap_values = None

    def fit(
        self,
        train_data: pd.DataFrame,
        train_target: pd.DataFrame,
        valid_data: pd.DataFrame = None,
        valid_target: pd.DataFrame = None,
    ) -> Tuple[lgb.Booster, pd.DataFrame, pd.Series]:
        """Fit one LGBM model."""
        # If valid data is not provided, then will be extracted out of train as 30%
        # For gradient boosting algorithms it is necessary to have valid data for
        # stooping of learning.
        if valid_data is None:
            train_data, valid_data, train_target, valid_target = train_test_split(
                train_data, train_target, test_size=0.3
            )

        dtrain = lgb.Dataset(train_data, label=train_target)
        dvalid = lgb.Dataset(valid_data, label=valid_target)

        booster = lgb.train(params=self.params, train_set=dtrain, valid_sets=dvalid, verbose_eval=20000)

        return booster, valid_data, valid_target

    def predict(self, test_data: pd.DataFrame, model) -> List[float]:
        """Predict model."""
        return model.predict(test_data)

    def fit_cv(self, train_data: pd.DataFrame, train_target: pd.DataFrame, n_splits=4) -> List[lgb.Booster]:
        """Fit cross validation model."""
        cross_val = KFold(n_splits=n_splits, shuffle=True)

        models = []

        for train_indexes, valid_indexes in cross_val.split(train_data):
            cv_train = train_data.iloc[train_indexes]
            cv_valid = train_data.iloc[valid_indexes]

            cv_train_target = train_target.iloc[train_indexes]
            cv_valid_target = train_target.iloc[valid_indexes]

            model = self.fit(cv_train, cv_train_target, cv_valid, cv_valid_target)

            models.append(model)

            print(f"AUC from LGBM on training data: {roc_auc_score(cv_train_target, model.predict(cv_train))}")
            print(f"AUC from LGBM on validation data: {roc_auc_score(cv_valid_target, model.predict(cv_valid))}")

        return models

    def predict_cv(self, test_data: pd.DataFrame, models: List[lgb.Booster]):
        """Predict cv model."""
        predictions = np.zeros(len(test_data))
        for _model in models:
            predictions += _model.predict(test_data) / len(models)
        return predictions

    def comp_var_imp(self, models: List[lgb.Booster], preds: List[str]) -> pd.DataFrame:
        """Compute variable importance of features."""
        importance_df = pd.DataFrame()
        importance_df["Feature"] = preds
        importance_df["Importance_gain"] = 0

        for _model in models:
            importance_df["Importance_gain"] = importance_df["Importance_gain"] + _model.feature_importance(
                importance_type="gain"
            ) / len(models)

        return importance_df

    def plot_importance(self, models: List[lgb.Booster], preds: List[str]):
        """Plot variable importances."""
        dataframe = self.comp_var_imp(models, preds)
        plt.figure(figsize=(20, len(preds) / 2))
        sns.barplot(
            x="Importance_gain",
            y="Feature",
            data=dataframe.sort_values(by="Importance_gain", ascending=False).head(len(preds)),
        )

    def print_shap_values(self, model: lgb.Booster, data: pd.DataFrame, ret: bool = False):
        """Compute SHAP values of the model for the data."""
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)

        if isinstance(shap_values, list):
            self.shap_values = shap_values[1]
        else:
            self.shap_values = shap_values

        shap.summary_plot(self.shap_values, data)
        shap.summary_plot(self.shap_values, data, plot_type="bar")

        if ret:
            return self.shap_values, explainer

    def shap_dependence_plot(self, data: pd.DataFrame, column: str, interaction_column: str = None):
        """Plot SHAP dependence plot."""
        if interaction_column:
            shap.dependence_plot(column, self.shap_values, data, interaction_index=interaction_column)
        else:
            shap.dependence_plot(column, self.shap_values, data)

