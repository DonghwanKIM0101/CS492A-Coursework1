from scipy import io
from sklearn.model_selection import train_test_split
import numpy as np
import random

def split(total_data, test_number):
    x = total_data['X']
    y = total_data['l']

    random.seed(10)

    te_idx = np.array([])
    for i in range(0,52):
        rand = random.sample(range(0,10), test_number)
        for j in range(test_number):
            te_idx = np.append(te_idx, [10*i + rand[j]])
    te_idx = te_idx.astype(int)

    tn_idx = np.array([])
    for i in range(0, 520):
        if i not in te_idx:
            tn_idx = np.append(tn_idx, i)

    tn_idx = tn_idx.astype(int)

    x_tn = x[:,tn_idx]
    y_tn = y[:,tn_idx]

    x_te = x[:,te_idx]
    y_te = y[:,te_idx]

    return (x_tn, y_tn, x_te, y_te)