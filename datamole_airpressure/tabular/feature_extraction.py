"""Class for feature extraction from measurements data."""
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


class FeatureExtractor:
    """
    Class for feature computation.

    Each method is supposed to calculate single statistical feature that
    can be afterward used in predictive model.
    """

    def __init__(self, config: dict):
        """Load config file."""
        self.config = config

    # STANDARD FEATURES
    @staticmethod
    def _extract_mean(sensor_measurements: list):
        """Extract Average of measurements."""
        return np.nanmean(sensor_measurements["Pressure"])

    @staticmethod
    def _extract_median(sensor_measurements: list):
        """Extract Median of measurements."""
        return np.nanmedian(sensor_measurements["Pressure"])

    @staticmethod
    def _extract_max(sensor_measurements: list):
        """Extract max of measurements."""
        return np.nanmax(sensor_measurements["Pressure"])

    @staticmethod
    def _extract_min(sensor_measurements: list):
        """Extract min of measurements."""
        return np.nanmin(sensor_measurements["Pressure"])

    @staticmethod
    def _extract_kurtosis(sensor_measurements: list):
        """Extract kurtosis of measurements."""
        return kurtosis(sensor_measurements["Pressure"], nan_policy="omit")

    @staticmethod
    def _extract_skew(sensor_measurements: list):
        """Extract skew of measurements."""
        return skew(sensor_measurements["Pressure"], nan_policy="omit")

    @staticmethod
    def _extract_max_min_diff(sensor_measurements: list):
        """Extract difference between max minus min."""
        return np.nanmax(sensor_measurements["Pressure"]) - np.nanmin(sensor_measurements["Pressure"])

    @staticmethod
    def _extract_sum(sensor_measurements: list):
        """Extract sum of measurements."""
        return np.nansum(sensor_measurements["Pressure"])

    # LAGGED FEATURES
    @staticmethod
    def _extract_mean_lagged(sensor_measurements: list):
        """Extract Average of measurements differences."""
        return np.nanmean(pd.Series(sensor_measurements["Pressure"]).diff())

    @staticmethod
    def _extract_median_lagged(sensor_measurements: list):
        """Extract Median of measurements differences."""
        return np.nanmedian(pd.Series(sensor_measurements["Pressure"]).diff())

    @staticmethod
    def _extract_max_lagged(sensor_measurements: list):
        """Extract max of measurements differences."""
        return np.nanmax(pd.Series(sensor_measurements["Pressure"]).diff())

    @staticmethod
    def _extract_min_lagged(sensor_measurements: list):
        """Extract min of measurements differences."""
        return np.nanmin(pd.Series(sensor_measurements["Pressure"]).diff())

    @staticmethod
    def _extract_kurtosis_lagged(sensor_measurements: list):
        """Extract kurtosis of measurements differences."""
        return kurtosis(pd.Series(sensor_measurements["Pressure"]).diff(), nan_policy="omit")

    @staticmethod
    def _extract_skew_lagged(sensor_measurements: list):
        """Extract skew of measurements differences."""
        return skew(pd.Series(sensor_measurements["Pressure"]).diff(), nan_policy="omit")

    @staticmethod
    def _extract_max_min_diff_lagged(sensor_measurements: list):
        """Extract difference between max and min of differences."""
        x = pd.Series(sensor_measurements["Pressure"]).diff()
        return np.nanmax(x) - np.nanmin(x)

    @staticmethod
    def _extract_sum_lagged(sensor_measurements: list):
        """Extract sum of measurements differences."""
        return np.nansum(pd.Series(sensor_measurements["Pressure"]).diff())

    # LAGGED FEATURES ABS
    @staticmethod
    def _extract_mean_lagged_abs(sensor_measurements: list):
        """Extract Average of abs measurements differences."""
        return np.nanmean(abs(pd.Series(sensor_measurements["Pressure"]).diff()))

    @staticmethod
    def _extract_median_lagged_abs(sensor_measurements: list):
        """Extract Median of abs measurements differences."""
        return np.nanmedian(abs(pd.Series(sensor_measurements["Pressure"]).diff()))

    @staticmethod
    def _extract_min_lagged_abs(sensor_measurements: list):
        """Extract min of abs measurements differences."""
        return np.nanmin(abs(pd.Series(sensor_measurements["Pressure"]).diff()))

    @staticmethod
    def _extract_kurtosis_lagged_abs(sensor_measurements: list):
        """Extract kurtosis of abs measurements differences."""
        return kurtosis(abs(pd.Series(sensor_measurements["Pressure"]).diff()), nan_policy="omit")

    @staticmethod
    def _extract_skew_lagged_abs(sensor_measurements: list):
        """Extract skew of abs measurements differences."""
        return skew(abs(pd.Series(sensor_measurements["Pressure"]).diff()), nan_policy="omit")

    @staticmethod
    def _extract_max_min_diff_lagged_abs(sensor_measurements: list):
        """Extract difference between max and min of abs differences."""
        x = abs(pd.Series(sensor_measurements["Pressure"]).diff())
        return np.nanmax(x) - np.nanmin(x)

    @staticmethod
    def _extract_sum_lagged_abs(sensor_measurements: list):
        """Extract sum of abs measurements differences."""
        return np.nansum(abs(pd.Series(sensor_measurements["Pressure"]).diff()))

    def run(self, data: pd.DataFrame):
        """Run extraction of the features according to the config."""
        # Base DF
        data_features = data.groupby(["MachineId", "MeasurementId"], as_index=False).count()
        data_features.drop(columns=["Pressure"], inplace=True)

        # Base DF for features that are extracted from non-zero measurements.
        data_features_non_zero = (
            data[data["Pressure"] > 0].groupby(["MachineId", "MeasurementId"], as_index=False).count()
        )
        data_features_non_zero.drop(columns=["Pressure"], inplace=True)

        grouped_base = data.groupby(["MachineId", "MeasurementId"])
        grouped_base_non_zero = data[data["Pressure"] > 0].groupby(["MachineId", "MeasurementId"])

        # STANDARD FEATURES
        if self.config["feature_mean"]:
            data_features["feature_mean"] = grouped_base.apply(self._extract_mean).tolist()

        if self.config["feature_median"]:
            data_features["feature_median"] = grouped_base.apply(self._extract_median).tolist()

        if self.config["feature_max"]:
            data_features["feature_max"] = grouped_base.apply(self._extract_max).tolist()

        if self.config["feature_min"]:
            data_features["feature_min"] = grouped_base.apply(self._extract_min).tolist()

        if self.config["feature_kurtosis"]:
            data_features["feature_kurtosis"] = grouped_base.apply(self._extract_kurtosis).tolist()

        if self.config["feature_skew"]:
            data_features["feature_skew"] = grouped_base.apply(self._extract_skew).tolist()

        if self.config["feature_max_min_diff"]:
            data_features["feature_max_min_diff"] = grouped_base.apply(self._extract_max_min_diff).tolist()

        if self.config["feature_sum"]:
            data_features["feature_sum"] = grouped_base.apply(self._extract_sum).tolist()

        # STANDARD NON-ZERO FEATURES
        if self.config["feature_mean_non_zero"]:
            data_features_non_zero["feature_mean_non_zero"] = grouped_base_non_zero.apply(self._extract_mean).tolist()

        if self.config["feature_median_non_zero"]:
            data_features_non_zero["feature_median_non_zero"] = grouped_base_non_zero.apply(
                self._extract_median
            ).tolist()

        if self.config["feature_max_non_zero"]:
            data_features_non_zero["feature_max_non_zero"] = grouped_base_non_zero.apply(self._extract_max).tolist()

        if self.config["feature_min_non_zero"]:
            data_features_non_zero["feature_min_non_zero"] = grouped_base_non_zero.apply(self._extract_min).tolist()

        if self.config["feature_kurtosis_non_zero"]:
            data_features_non_zero["feature_kurtosis_non_zero"] = grouped_base_non_zero.apply(
                self._extract_kurtosis
            ).tolist()

        if self.config["feature_skew_non_zero"]:
            data_features_non_zero["feature_skew_non_zero"] = grouped_base_non_zero.apply(self._extract_skew).tolist()

        if self.config["feature_max_min_diff_non_zero"]:
            data_features_non_zero["feature_max_min_diff_non_zero"] = grouped_base_non_zero.apply(
                self._extract_max_min_diff
            ).tolist()

        if self.config["feature_sum_non_zero"]:
            data_features_non_zero["feature_sum_non_zero"] = grouped_base_non_zero.apply(self._extract_sum).tolist()

        # STANDARD LAGGED FEATURES
        if self.config["feature_mean_lagged"]:
            data_features["feature_mean_lagged"] = grouped_base.apply(self._extract_mean_lagged).tolist()

        if self.config["feature_median_lagged"]:
            data_features["feature_median_lagged"] = grouped_base.apply(self._extract_median_lagged).tolist()

        if self.config["feature_max_lagged"]:
            data_features["feature_max_lagged"] = grouped_base.apply(self._extract_max_lagged).tolist()

        if self.config["feature_min_lagged"]:
            data_features["feature_min_lagged"] = grouped_base.apply(self._extract_min_lagged).tolist()

        if self.config["feature_kurtosis_lagged"]:
            data_features["feature_kurtosis_lagged"] = grouped_base.apply(self._extract_kurtosis_lagged).tolist()

        if self.config["feature_skew_lagged"]:
            data_features["feature_skew_lagged"] = grouped_base.apply(self._extract_skew_lagged).tolist()

        if self.config["feature_max_min_diff_lagged"]:
            data_features["feature_max_min_diff_lagged"] = grouped_base.apply(
                self._extract_max_min_diff_lagged
            ).tolist()

        if self.config["feature_sum_lagged"]:
            data_features["feature_sum_lagged"] = grouped_base.apply(self._extract_sum_lagged).tolist()

        # STANDARD ABS LAGGED FEATURES
        if self.config["feature_mean_lagged_abs"]:
            data_features["feature_mean_lagged_abs"] = grouped_base.apply(self._extract_mean_lagged_abs).tolist()

        if self.config["feature_median_lagged_abs"]:
            data_features["feature_median_lagged_abs"] = grouped_base.apply(self._extract_median_lagged_abs).tolist()

        if self.config["feature_min_lagged_abs"]:
            data_features["feature_min_lagged_abs"] = grouped_base.apply(self._extract_min_lagged_abs).tolist()

        if self.config["feature_kurtosis_lagged_abs"]:
            data_features["feature_kurtosis_lagged_abs"] = grouped_base.apply(
                self._extract_kurtosis_lagged_abs
            ).tolist()

        if self.config["feature_skew_lagged_abs"]:
            data_features["feature_skew_lagged_abs"] = grouped_base.apply(self._extract_skew_lagged_abs).tolist()

        if self.config["feature_max_min_diff_lagged_abs"]:
            data_features["feature_max_min_diff_lagged_abs"] = grouped_base.apply(
                self._extract_max_min_diff_lagged_abs
            ).tolist()

        if self.config["feature_sum_lagged_abs"]:
            data_features["feature_sum_lagged_abs"] = grouped_base.apply(self._extract_sum_lagged_abs).tolist()

        return pd.merge(
            data_features,
            data_features_non_zero,
            how="left",
            left_on=["MachineId", "MeasurementId"],
            right_on=["MachineId", "MeasurementId"],
        )
