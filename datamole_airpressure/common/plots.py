"""Script with auxiliary functions for plotting."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from datamole_airpressure.common.metrics import individual_auc


def visualise_feature(data, col, target_col):
    """Print separated distributions of the feature according the target."""
    subset = data[((data[col].notnull()) & (data[target_col].notnull()))]
    sns.displot(data=subset, x=col, hue=target_col, kind='kde', common_norm=False, fill=True, height=5, aspect=1.5)
    plt.xlim(np.quantile(subset[col], 0.01), np.quantile(subset[col], 0.99))
    plt.show()
    auc = individual_auc(subset, col, target_col)
    print(auc)


def show_series(data: pd.DataFrame, labels: pd.DataFrame, machine_id: str, measurement_id: int):
    """Plot one cycle as timeseries of the measurement."""
    print(f"machine_id: {machine_id}, measurement_id:{measurement_id}")
    print("No observatons:")
    print(data[((data["MachineId"] == machine_id) & (data["MeasurementId"] == measurement_id))]["Pressure"].shape)
    print(labels[((labels["MachineId"] == machine_id) & (labels["MeasurementId"] == measurement_id))])
    plt.plot(data[((data["MachineId"] == machine_id) & (data["MeasurementId"] == measurement_id))]["Pressure"])


def show_series(data: pd.DataFrame, labels: pd.DataFrame, machine_id: str, measurement_id: int):
    """Plot one cycle as timeseries of the measurement."""
    print(f"machine_id: {machine_id}, measurement_id:{measurement_id}")
    print("No observatons:")
    print(data[((data["MachineId"] == machine_id) & (data["MeasurementId"] == measurement_id))]["Pressure"].shape)
    print(labels[((labels["MachineId"] == machine_id) & (labels["MeasurementId"] == measurement_id))])
    plt.plot(data[((data["MachineId"] == machine_id) & (data["MeasurementId"] == measurement_id))]["Pressure"])