"""Class for the computer vision model."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import tqdm
from sklearn.metrics import log_loss, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datamole_airpressure.computer_vision.dataset import TorchDataset

TRAIN_SET_PORTION = 0.7
NUM_TARGET_LABELS = 2
NUM_DATALOADER_WORKERS = 8
OPTIMIZER_LEARNING_RATE_PARAM = 1e-6
OPTIMIZER_EPS_PARAM_PARAM = 1e-8


class VisionModel:
    """Class for Deep learning model."""

    def __init__(self, config: dict):
        """Initialize model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = config["num_epochs"]
        self.model = self.define_model()
        self.model = self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=OPTIMIZER_LEARNING_RATE_PARAM, eps=OPTIMIZER_EPS_PARAM_PARAM)
        self.loss_function = nn.CrossEntropyLoss().cuda()
        self.train_batch_size = config["train_batch_size"]
        self.test_batch_size = config["test_batch_size"]
        self.model_results = {}
        self.softmax_fn = nn.Softmax(dim=1)

    def define_model(self):
        """Return NN architecture."""
        model_ft = models.resnet50(pretrained=True)

        # Define classification head of the model
        # Input size is #model_ft.fc.in_features (1024 for resnet50)
        # Output size is NUM_TARGET_LABELS (2 in this case)
        model_ft.fc = nn.Sequential(
            nn.Linear(model_ft.fc.in_features, NUM_TARGET_LABELS),
        )

        return model_ft

    def fit(self, labels: pd.DataFrame, ret_results: bool = False):
        """Fit model."""
        # Split train set to train/valid sets.
        labels["random"] = np.random.random(size=len(labels))
        labels_train = labels[labels["random"] <= TRAIN_SET_PORTION].reset_index(drop=True)
        labels_valid = labels[labels["random"] > TRAIN_SET_PORTION].reset_index(drop=True)

        # Tranform data to pytorch-mandatory dataset class
        train_data = TorchDataset(labels_train)
        train_loader = DataLoader(
            train_data, batch_size=self.train_batch_size, shuffle=True, num_workers=NUM_DATALOADER_WORKERS
        )

        valid_data = TorchDataset(labels_valid)
        valid_loader = DataLoader(
            valid_data, batch_size=self.train_batch_size, shuffle=False, num_workers=NUM_DATALOADER_WORKERS
        )

        # Train model
        for epoch in range(self.num_epochs):

            self.model_results[f"epoch_{epoch}"] = {}
            predictions_all_train = []
            labels_all_train = []
            predictions_all_valid = []
            labels_all_valid = []

            self.model.train()  # Turn on training mode
            for batch in tqdm.tqdm(train_loader):

                self.optimizer.zero_grad()  # Erase gradient

                inputs = batch[0].to(self.device, dtype=torch.float)
                targets = batch[1].to(self.device, dtype=torch.float)

                predictions = self.model(inputs)

                loss = self.loss_function(predictions, targets)
                # Calculate gradient of the loss with respect to weights
                loss.backward()

                # Update weights of the model
                self.optimizer.step()

                predictions_all_train = predictions_all_train + self.softmax_fn(predictions)[:, 1].detach().tolist()
                labels_all_train = labels_all_train + targets[:, 1].detach().tolist()

            # Validation stage
            # DROPOUT is not applied
            with torch.no_grad():

                for batch in valid_loader:
                    inputs = batch[0].to(self.device, dtype=torch.float)
                    targets = batch[1].to(self.device, dtype=torch.float)

                    predictions = self.model(inputs)

                    predictions_all_valid = predictions_all_valid + self.softmax_fn(predictions)[:, 1].detach().tolist()
                    labels_all_valid = labels_all_valid + targets[:, 1].detach().tolist()

        print(f"Results for epoch {epoch}")
        print("Train auc:")
        print(roc_auc_score(labels_all_train, predictions_all_train))
        print("Valid auc:")
        print(roc_auc_score(labels_all_valid, predictions_all_valid))
        self.model_results[f"epoch_{epoch}"]["train_target"] = labels_all_train
        self.model_results[f"epoch_{epoch}"]["train_predictions"] = predictions_all_train
        self.model_results[f"epoch_{epoch}"]["valid_target"] = labels_all_valid
        self.model_results[f"epoch_{epoch}"]["valid_predictions"] = predictions_all_valid

        self.visualise_results()

        if ret_results:
            return self.model_results

    def predict(self, labels):
        """
        Predict method.

        Return tuple of predicted probabilities, predicted labels and target
        """
        test_data = TorchDataset(labels)
        test_loader = DataLoader(test_data, batch_size=self.test_batch_size, shuffle=False, num_workers=8)

        predictions = []
        predictions_labels = []
        labels = []

        self.model.eval()

        with torch.no_grad():

            for batch in tqdm.tqdm(test_loader):
                inputs = batch[0].to(self.device, dtype=torch.float)
                targets = batch[1].to(self.device, dtype=torch.float)

                outputs = self.model(inputs)

                predictions = predictions + self.softmax_fn(outputs)[:, 1].detach().tolist()
                labels = labels + targets[:, 1].detach().tolist()
                predictions_labels = predictions_labels + outputs.argmax(dim=1).detach().tolist()

        return predictions, predictions_labels, labels

    def visualise_results(self):
        """Plot learning curves for LOGLOSS and AUC."""
        train_loss = []
        valid_loss = []
        train_auc = []
        valid_auc = []

        for epoch in range(self.num_epochs):
            train_auc.append(
                roc_auc_score(
                    self.model_results[f"epoch_{epoch}"]["train_target"],
                    self.model_results[f"epoch_{epoch}"]["train_predictions"],
                )
            )
            valid_auc.append(
                roc_auc_score(
                    self.model_results[f"epoch_{epoch}"]["valid_target"],
                    self.model_results[f"epoch_{epoch}"]["valid_predictions"],
                )
            )

            train_loss.append(
                log_loss(
                    self.model_results[f"epoch_{epoch}"]["train_target"],
                    self.model_results[f"epoch_{epoch}"]["train_predictions"],
                )
            )
            valid_loss.append(
                log_loss(
                    self.model_results[f"epoch_{epoch}"]["valid_target"],
                    self.model_results[f"epoch_{epoch}"]["valid_predictions"],
                )
            )

        fig, axs = plt.subplots(2, figsize=(14, 8))
        axs[0].plot(train_loss)
        axs[0].plot(valid_loss)
        axs[0].set_title("blue = Train Loss, orange = Valid Loss")

        axs[1].plot(train_auc)
        axs[1].plot(valid_auc)
        axs[1].set_title("Blue = Train AUC, orange = Valid AUC")
        plt.show()
