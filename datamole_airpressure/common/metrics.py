"""Script for performance metrics calculation calculation."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def individual_auc(data:pd.DataFrame, col:str, target_col: str)-> float:
    """Print individual auc of the column."""
    subset = data[data[col].notnull()]
    auc = roc_auc_score(subset[target_col].astype("int"), subset[col])
    if auc < 0.5: # In case that the predictor is negatively sorting target
        auc = 0.5 + (0.5-auc)
    return auc

class Metrics:
    """Class for computation of standard ML classification metrics."""

    def __init__(self, target: pd.Series, predictions: pd.Series, threshold: float = None):
        """Construct."""
        self.target = target
        self.predictions = predictions
        self.data = pd.DataFrame.from_dict({"target": target, "prediction": predictions})
        self.data = self.data.sort_values(by="prediction", ascending=False).reset_index(drop=True)
        self.threshold = threshold

    def calculate_stats(self):
        """
        Calculate Precision, Recall and F1 for all thresholds.

        Should be used as a helper function for optional threshold analysis and selection.
        This method shouldn't be used for test set, where threshold parameter should be passed as
        parameter to the constructoor."""
        self.data["Precision"] = None
        self.data["Recall"] = None
        self.data["F1"] = None

        count_TP_total = self.data["target"].sum()

        for index, row in self.data.iterrows():
            selected_subset = self.data[self.data["prediction"] >= row["prediction"]]
            selected_subset_TP_count = selected_subset["target"].sum()

            self.data["Precision"][index] = selected_subset_TP_count / len(selected_subset)
            self.data["Recall"][index] = selected_subset_TP_count / count_TP_total
            self.data["F1"][index] = (
                2
                * self.data["Precision"][index]
                * self.data["Recall"][index]
                / (self.data["Precision"][index] + self.data["Recall"][index])
            )

        max_F1 = np.max(self.data["F1"])
        self.threshold = self.data[self.data["F1"] == max_F1]["prediction"].iloc[0]

    def calculate_label_predictions(self):
        """Create column with predicted label, based on optimal_threshold."""
        assert self.threshold is not None
        self.data["Predicted_label"] = np.where(self.data["prediction"] > self.threshold, 1, 0)

    def plot_roc(self):
        """Plot ROC curve."""
        plt.figure(figsize=(8, 8))
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        fpr, tpr, _ = roc_curve(self.data["target"], self.data["prediction"])
        auc = roc_auc_score(self.data["target"], self.data["prediction"])
        plt.plot(fpr, tpr, label=f"ROC curve, auc={round(auc, 3)}")
        plt.legend(loc="lower right")
        plt.show()
        plt.close()

    def plot_F1_vs_threshold(self):
        """Plot F1 vs threshold."""
        plt.figure(figsize=(8, 8))
        plt.plot(
            self.data["prediction"],
            self.data["F1"],
            marker=".",
            label="F1 vs threshold",
        )
        plt.xlabel("Threshold")
        plt.ylabel("F1")
        plt.legend()
        plt.show()

    def plot_precision_vs_threshold(self):
        """Plot Precision vs threshold."""
        plt.figure(figsize=(8, 8))
        plt.plot(
            self.data["prediction"],
            self.data["Precision"],
            marker=".",
            label="Precision vs threshold",
        )
        plt.xlabel("Threshold")
        plt.ylabel("Precision")
        plt.legend()
        plt.show()

    def plot_recall_vs_threshold(self):
        """Plot Recall vs threshold."""
        plt.figure(figsize=(8, 8))
        plt.plot(
            self.data["prediction"],
            self.data["Recall"],
            marker=".",
            label="Recall vs threshold",
        )
        plt.xlabel("Threshold")
        plt.ylabel("Recall")
        plt.legend()
        plt.show()

    def print_all_metrics(self):
        """Print all metrics."""
        print("Area under ROC curve:")
        print(roc_auc_score(self.data["target"], self.data["prediction"]))
        print("F1 score:")
        print(f1_score(self.data["target"], self.data["Predicted_label"]))
        print("Precision")
        print(precision_score(self.data["target"], self.data["Predicted_label"]))
        print("Recall:")
        print(recall_score(self.data["target"], self.data["Predicted_label"]))
        print("Confusion Matrix:")
        print(confusion_matrix(self.data["target"], self.data["Predicted_label"]))
