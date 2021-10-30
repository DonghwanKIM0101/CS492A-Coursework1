from scipy import io
from sklearn.model_selection import train_test_split
import numpy as np
import random

def split_training_set(total_data, total_label, n_subset):
    x = total_data
    y = total_label
    total_index = np.arange(x.shape[1])

    data_subsets = []
    label_subsets = []

    for i in range(n_subset):
        index = (total_index % n_subset == i)
        data_subsets.append(x[:, index]) # data
        label_subsets.append(y[:, index]) # label


    # index = (total_index % 8 == 0)
    # data_subsets.append(x[:, index]) # data
    # label_subsets.append(y[:, index]) # label

    # index = (total_index % 8 == 1)
    # data_subsets.append(x[:, index]) # data
    # label_subsets.append(y[:, index]) # label

    # index = (total_index % 8 == 2) + (total_index % 8 == 3)
    # data_subsets.append(x[:, index]) # data
    # label_subsets.append(y[:, index]) # label

    # index = (total_index % 8 == 4) + (total_index % 8 == 5) + (total_index % 8 == 6) + (total_index % 8 == 7)
    # data_subsets.append(x[:, index]) # data
    # label_subsets.append(y[:, index]) # label

    return data_subsets, label_subsets