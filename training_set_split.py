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

    # #data
    # x_tn_1 = x[:, ((index%8 == 0) + index%8 == 1)]
    # x_tn_2 = x[:, ((index%8 == 2) + index%8 == 3)]
    # x_tn_3 = x[:, ((index%8 == 4) + index%8 == 5)]
    # x_tn_4 = x[:, ((index%8 == 6) + index%8 == 7)]

    # #label
    # y_tn_1 = y[:, ((index%8 == 0) + index%8 == 1)]
    # y_tn_2 = y[:, ((index%8 == 2) + index%8 == 3)]
    # y_tn_3 = y[:, ((index%8 == 4) + index%8 == 5)]
    # y_tn_4 = y[:, ((index%8 == 6) + index%8 == 7)]

    return data_subsets, label_subsets