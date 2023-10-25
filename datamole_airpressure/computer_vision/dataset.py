"""Data handling functions and classes."""
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from torch.utils.data import Dataset
import cv2

DESIRED_IMAGE_WIDTH = 496
DESIRED_IMAGE_HEIGHT = 369


def save_images(labels: pd.DataFrame, data: pd.DataFrame, path_to_save: str):
    """Save measurements as png images."""
    # Iterate through all machine ids and measurement ids from labels dataframe.
    for index, row in tqdm.tqdm(labels.iterrows()):
        mchn_id = row["MachineId"]
        meas_id = row["MeasurementId"]
        filename = f"{mchn_id}_{meas_id}.png"
        filepath = os.path.join(path_to_save, filename)
        if os.path.isfile(filepath):  # Skip if file already exists
            continue
        else:
            measurements = data[((data["MachineId"] == mchn_id) & (data["MeasurementId"] == meas_id))]["Pressure"]
            plt.plot(measurements)
            # Save only pure image without axis
            plt.axis("off")
            plt.savefig(filepath, bbox_inches="tight", pad_inches=0)
            plt.close()


class TorchDataset(Dataset):
    """Mandatory Pytorch-structured class for data handling."""

    def __init__(self, labels: pd.DataFrame, images_folder_path: str):
        """
        Construct Dataset.

        Usually, all the images are loaded to an array, from which can be
        afterward loaded with __getitem__ method. Unfortunately, with current
        amount of images and lower capacity of RAM, instead of saving of images
        to the array, only paths to the images are saved and real images are
        loaded in the moment when they are actually needed during the training
        method when batch of the input images are fed into the model.
        """
        self.filepaths = []  # Array of all
        self.labels = []  # Output array with targets

        for index, row in tqdm.tqdm(labels.iterrows()):
            mchn_id = row["MachineId"]
            meas_id = row["MeasurementId"]
            label = int(row["PumpFailed"])
            filename = f"{mchn_id}_{meas_id}.png"
            filepath = os.path.join(images_folder_path, filename)
            self.filepaths.append(filepath)
            label_array = np.zeros(2)
            label_array[label] = 1
            self.labels.append(label_array)

    @staticmethod
    def _read_and_preprocess_image_from_path(filepath: str):
        """
        Read image from filepath and create from that numpy array.

        Because of the fact that not all the images are of the same
        shape (number of measurements was not always the same for
        different measurements ids), it is necessary to resize the images
        to the same shape, which is chosen to 496 x 369, because it is
        most frequent shape of the images.
        """
        # Read image and rescale pixels values to 0-1 range.
        image_orig = imageio.imread(filepath) / 255
        # Take only first axis
        image_orig = image_orig[:, :, 0]
        image_orig = cv2.resize(image_orig, ((DESIRED_IMAGE_WIDTH, DESIRED_IMAGE_HEIGHT)))

        # Moddel that will be used expect images with depth 3
        image = np.stack((image_orig, image_orig, image_orig))
        return image

    def __getitem__(self, index: int):
        """Mandatory method, return image and target at index-position."""
        return self._read_and_preprocess_image_from_path(self.filepaths[index]), self.labels[index]

    def __len__(self):
        """Mandatory method. Len of data."""
        return len(self.filepaths)
