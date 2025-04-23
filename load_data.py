# -*- coding: utf-8 -*-
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
