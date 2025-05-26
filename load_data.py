# -*- coding: utf-8 -*-
"""
This script defines the `load_data` function for preprocessing vibration signal data 
from MATLAB `.mat` files, specifically tailored for fault diagnosis tasks.

Main functionalities:
1. Load time-series data from a `.mat` file under the given key (`temp`).
2. Remove samples from specified classes (`mis_class`) to allow flexible class selection.
3. Split the remaining data into training and testing sets based on the given number of samples per class.
4. Apply Fast Fourier Transform (FFT) to convert time-domain signals into frequency-domain features.
5. Retain only the first half of the frequency spectrum (due to symmetry).
6. Convert the resulting data into PyTorch `FloatTensor` format for use in deep learning models.

Function:
    load_data(path, temp, com_class, class_sample, mis_class)

Parameters:
    - path (str): File path to the `.mat` data file.
    - temp (str): Key used to extract the dataset from the loaded `.mat` dictionary.
    - com_class (int): Total number of original classes before excluding any.
    - class_sample (int): Number of training samples per class.
    - mis_class (list): List of class labels to exclude from the dataset.

Returns:
    - train_x (FloatTensor): Training features in frequency domain.
    - train_y (FloatTensor): Training labels.
    - test_x (FloatTensor): Testing features in frequency domain.
    - test_y (FloatTensor): Testing labels.
"""

import torch
import scipy.io as scio
from scipy.fftpack import fft
import numpy as np

def load_data(path, temp, com_class, class_sample, mis_class):
    data_temp = scio.loadmat(path)
    data = data_temp.get(temp)
    n_class = com_class - len(mis_class)

    for i_c in mis_class:
        mis_class_id = np.argwhere(data[:, -1] == i_c)
        data = np.delete(data, mis_class_id, axis=0)

    train_x = data[:class_sample * n_class, :, :-1]
    test_x  = data[class_sample * n_class:, :, :-1]

    # FFT
    train_x = 2 * abs(fft(train_x).real) / 2048
    train_x = train_x[:, :, :1024]
    test_x = 2 * abs(fft(test_x).real) / 2048
    test_x = test_x[:, :, :1024]

    train_x = torch.FloatTensor(train_x)
    train_y = torch.FloatTensor(data[:class_sample * n_class, :, -1])
    test_x = torch.FloatTensor(test_x)
    test_y = torch.FloatTensor(data[class_sample * n_class:, :, -1])

    return train_x, train_y, test_x, test_y
