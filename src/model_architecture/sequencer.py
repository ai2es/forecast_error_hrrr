import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class SequenceDatasetMultiTask(Dataset):
    """Dataset class for multi-task learning with station-specific data."""

    def __init__(
        self,
        dataframe,
        target,
        features,
        sequence_length,
        forecast_steps,
        device,
        metvar,
    ):
        self.dataframe = dataframe
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.forecast_steps = forecast_steps
        self.device = device
        self.metvar = metvar
        self.y = torch.tensor(dataframe[target].values).float().to(device)
        self.X = torch.tensor(dataframe[features].values).float().to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x_start = i
        x_end = i + (self.sequence_length + self.forecast_steps)
        y_start = i + self.sequence_length
        y_end = y_start + self.forecast_steps
        x = self.X[x_start:x_end, :]
        y = self.y[y_start:y_end].unsqueeze(1)

        if x.shape[0] < (self.sequence_length + self.forecast_steps):
            _x = torch.zeros(
                (
                    (self.sequence_length + self.forecast_steps) - x.shape[0],
                    self.X.shape[1],
                ),
                device=self.device,
            )
            x = torch.cat((x, _x), 0)

        if y.shape[0] < self.forecast_steps:
            _y = torch.zeros((self.forecast_steps - y.shape[0], 1), device=self.device)
            y = torch.cat((y, _y), 0)

        x[-self.forecast_steps :, -int(4 * 16) :] = x[
            -int(self.forecast_steps + 1), -int(4 * 16) :
        ].clone()

        return x, y
