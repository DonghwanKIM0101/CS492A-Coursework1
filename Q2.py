import scipy.io
import scipy.linalg
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from train_test_split import split
from training_set_split import split_training_set

M = 416
INDEX = 8 # image to reconstruct
WIDTH = 46
HEIGHT = 56
TEST_NUMBER = 2

def solve_eig(S):
    eig_val, eig_vec = np.linalg.eigh(S)

    sorted_eig_val = np.flip(eig_val)
    sorted_eig_vec = np.flip(eig_vec, axis=1)

    positive_map = sorted_eig_val > 0
    positive_sorted_eig_val = sorted_eig_val[positive_map]
    positive_sorted_eig_vec = sorted_eig_vec[:,positive_map]

    return positive_sorted_eig_val, positive_sorted_eig_vec


def incremental_pca(mean_1, N_1, S_1, P_1, batch_2):
    mean_2 = np.average(batch_2,1)

    A_2 = np.subtract(batch_2, mean_2.reshape((-1,1)))
    N_2 = batch_2.shape[1]
    N = N_1 + N_2
    S_2 = np.matmul(A_2, A_2.transpose()) / N_2
    lambda_2, P_2 = solve_eig(S_2)

    mean = (mean_1 * N_1 + mean_2 * N_2) / N
    S = N_1 / N * S_1 + N_2 / N * S_2 + N_1 * N_2 / (N ** 2) * np.matmul((mean_1 - mean_2), (mean_1 - mean_2).transpose())


    # test = np.concatenate((P_1, P_2, (mean_1 - mean_2).reshape(-1,1)), axis=1)
    # print(test.shape)
    q, r = np.linalg.qr(np.concatenate((P_1, P_2, (mean_1 - mean_2).reshape(-1,1)), axis=1))
    # q = scipy.linalg.orth(np.concatenate((P_1, P_2, (mean_1 - mean_2).reshape(-1,1)), axis=1))

    eig_val, eig_vec = solve_eig(S)

    plt.plot(eig_val)
    plt.show()

    result_S = np.matmul(np.matmul(q.transpose(), S), q)
    eig_val, eig_vec = solve_eig(result_S)
    eig_vec = np.matmul(q, eig_vec)

    plt.plot(eig_val)
    plt.show()

    return eig_val, eig_vec, mean, N, S


mat = scipy.io.loadmat('face.mat')
data_train, label_train , data_test, label_test = split(mat, TEST_NUMBER)
x_tn_1, x_tn_2, x_tn_3, x_tn_4, y_tn_1, y_tn_2, y_tn_3, y_tn_4 = split_training_set(data_train, label_train)


mean_1 = np.average(x_tn_1,1)
A_1 = np.subtract(x_tn_1, mean_1.reshape((-1,1)))
N_1 = x_tn_1.shape[1]
S_1 = np.matmul(A_1, A_1.transpose()) / N_1
lambda_1, P_1 = solve_eig(S_1)

eig_val, eig_vec, mean, N, S = incremental_pca(mean_1, N_1, S_1, P_1, x_tn_2)

# eig_val, eig_vec, mean, N, S = incremental_pca(mean, N, S, eig_vec, x_tn_3)

# eig_val, eig_vec, mean, N, S = incremental_pca(mean, N, S, eig_vec, x_tn_4)

# print(N)
# plt.plot(eig_val)
# plt.show()



# data = data_train.reshape((WIDTH,HEIGHT,-1))
# data = np.transpose(data, (1,0,2))

# mean_image = np.uint8(np.average(data,2))
# mean_flatten = np.average(data_train,1)
# # cv2.imwrite('mean_image.jpg', np.uint8(mean_image))

# A = np.subtract(data_train, mean_flatten.reshape((-1,1)))
# S = np.matmul(A, A.transpose()) / data_train.shape[1]
# S_low = np.matmul(A.transpose(), A) / data_train.shape[1]

# eig_val, eig_vec = solve_eig(S)
# eig_val_low, eig_vec_low = solve_eig(S_low)


# # Plot eigen values.
print(eig_vec.shape)
# print(eig_vec_low.shape)
plt.plot(eig_val)
# plt.plot(eig_val_low)
plt.show()

