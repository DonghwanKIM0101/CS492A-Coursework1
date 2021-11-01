import scipy.io
import scipy.linalg
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from train_test_split import split
from training_set_split import split_training_set

M = 100
INDEX = 8 # image to reconstruct
WIDTH = 46
HEIGHT = 56
TEST_NUMBER = 2
ELAPSED_TIME = 0
ELAPSED_TIME_INCREMENTAL = 0
N_SUBSETS = 4

def solve_eig(S):
    global ELAPSED_TIME
    start = time.time()
    eig_val, eig_vec = np.linalg.eigh(S)
    ELAPSED_TIME += time.time() - start

    sorted_eig_val = np.flip(eig_val)
    sorted_eig_vec = np.flip(eig_vec, axis=1)

    positive_map = sorted_eig_val > 0
    positive_sorted_eig_val = sorted_eig_val[positive_map]
    positive_sorted_eig_vec = sorted_eig_vec[:,positive_map]

    return positive_sorted_eig_val, positive_sorted_eig_vec


def incremental_pca(mean_1, N_1, S_1, P_1, batch_2):
    global ELAPSED_TIME_INCREMENTAL

    mean_2 = np.average(batch_2,1)

    A_2 = np.subtract(batch_2, mean_2.reshape(-1,1))
    N_2 = batch_2.shape[1]
    N = N_1 + N_2
    S_2 = np.matmul(A_2, A_2.transpose()) / N_2
    lambda_2, P_2 = solve_eig(S_2)

    start = time.time()
    mean = (mean_1 * N_1 + mean_2 * N_2) / N
    S = N_1 / N * S_1 + N_2 / N * S_2 + N_1 * N_2 / (N ** 2) * np.matmul((mean_1 - mean_2).reshape(-1,1), (mean_1 - mean_2).reshape(-1,1).transpose())

    q, r = np.linalg.qr(np.concatenate((P_1, P_2, (mean_1 - mean_2).reshape(-1,1)), axis=1))

    result_S = np.matmul(np.matmul(q.transpose(), S), q)
    ELAPSED_TIME_INCREMENTAL += time.time() - start

    eig_val, eig_vec = solve_eig(result_S)

    start = time.time()
    eig_vec = np.matmul(q, eig_vec)
    ELAPSED_TIME_INCREMENTAL += time.time() - start

    return eig_val, eig_vec, mean, N, S


mat = scipy.io.loadmat('face.mat')
data_train, label_train , data_test, label_test = split(mat, TEST_NUMBER)
data_subsets, label_subsets = split_training_set(data_train, label_train, N_SUBSETS)

# incremental PCA
mean = np.average(data_subsets[0],1)
A = np.subtract(data_subsets[0], mean.reshape(-1,1))
N = data_subsets[0].shape[1]
S = np.matmul(A, A.transpose()) / N
eig_val, eig_vec = solve_eig(S)

for i in range(1,N_SUBSETS):
    eig_val, eig_vec, mean, N, S = incremental_pca(mean, N, S, eig_vec, data_subsets[i])

A = np.subtract(data_train, mean.reshape(-1,1))


# # batch PCA
# data = data_train
# mean_batch = np.average(data,1)
# A = np.subtract(data, mean_batch.reshape(-1,1))
# S = np.matmul(A, A.transpose()) / data.shape[1]
# eig_val_batch, eig_vec_batch = solve_eig(S)

# # PCA with only first subset
# data = data_subsets[0]
# mean_first = np.average(data,1)
# A = np.subtract(data, mean_first.reshape(-1,1))
# S = np.matmul(A, A.transpose()) / data.shape[1]
# eig_val_first, eig_vec_first = solve_eig(S)


print("elasped time is ", ELAPSED_TIME)
print("incremental time is ", ELAPSED_TIME_INCREMENTAL)

# # Plot eigen values.
# # print(eig_vec.shape)
# # print(eig_vec_low.shape)
# plt.plot(eig_val)
# plt.plot(eig_val_batch)
# plt.plot(eig_val_first)
# plt.show()


# # face recognition accuracy
# Accuracies = []
# Accuracies_batch = []
# Accuracies_first = []

# for m in tqdm(range(eig_val.shape[0])):
#     weight = np.matmul(A.transpose(), eig_vec[:,:m])

#     A_test = np.subtract(data_test, mean.reshape(-1,1))
#     weight_test = np.matmul(A_test.transpose(), eig_vec[:,:m])

#     count = 0
#     for i, test in enumerate(weight_test):
#         error = np.subtract(test.reshape(1,-1), weight)
        
#         error = np.linalg.norm(error, axis=1)
#         count += int(label_train[:,np.argmin(error)] == label_test[:,i])

#     Accuracies.append(count / weight_test.shape[0])

# for m in tqdm(range(eig_val_batch.shape[0])):
#     weight = np.matmul(A.transpose(), eig_vec_batch[:,:m])

#     A_test = np.subtract(data_test, mean_batch.reshape(-1,1))
#     weight_test = np.matmul(A_test.transpose(), eig_vec_batch[:,:m])

#     count = 0
#     for i, test in enumerate(weight_test):
#         error = np.subtract(test.reshape(1,-1), weight)
        
#         error = np.linalg.norm(error, axis=1)
#         count += int(label_train[:,np.argmin(error)] == label_test[:,i])

#     Accuracies_batch.append(count / weight_test.shape[0])

# for m in tqdm(range(eig_val_first.shape[0])):
#     weight = np.matmul(A.transpose(), eig_vec_first[:,:m])

#     A_test = np.subtract(data_test, mean_first.reshape(-1,1))
#     weight_test = np.matmul(A_test.transpose(), eig_vec_first[:,:m])

#     count = 0
#     for i, test in enumerate(weight_test):
#         error = np.subtract(test.reshape(1,-1), weight)
        
#         error = np.linalg.norm(error, axis=1)
#         count += int(label_train[:,np.argmin(error)] == label_test[:,i])

#     Accuracies_first.append(count / weight_test.shape[0])

# plt.plot(Accuracies)
# plt.plot(Accuracies_batch)
# plt.plot(Accuracies_first)
# plt.show()


# # Quantitatively face reconstruction
# Errors = []
# Errors_batch = []
# Errors_first = []

# for m in tqdm(range(eig_val.shape[0])):
#     error = 0

#     for index in range(data_test.shape[1]):
#         phi = data_test[:,index] - mean

#         weight = np.matmul(phi.reshape(1,-1), eig_vec[:,:m])

#         face_recon = mean + np.matmul(eig_vec[:,:m], weight.transpose()).squeeze()

#         error += np.linalg.norm(face_recon - data_test[:,index])
    
#     error /= data_test.shape[1]
#     Errors.append(error)

# for m in tqdm(range(eig_val_batch.shape[0])):
#     error = 0

#     for index in range(data_test.shape[1]):
#         phi = data_test[:,index] - mean_batch

#         weight = np.matmul(phi.reshape(1,-1), eig_vec_batch[:,:m])

#         face_recon = mean_batch + np.matmul(eig_vec_batch[:,:m], weight.transpose()).squeeze()

#         error += np.linalg.norm(face_recon - data_test[:,index])
    
#     error /= data_test.shape[1]
#     Errors_batch.append(error)

# for m in tqdm(range(eig_val_first.shape[0])):
#     error = 0

#     for index in range(data_test.shape[1]):
#         phi = data_test[:,index] - mean_first

#         weight = np.matmul(phi.reshape(1,-1), eig_vec_first[:,:m])

#         face_recon = mean_first + np.matmul(eig_vec_first[:,:m], weight.transpose()).squeeze()

#         error += np.linalg.norm(face_recon - data_test[:,index])
    
#     error /= data_test.shape[1]
#     Errors_first.append(error)

# plt.plot(Errors)
# plt.plot(Errors_batch)
# plt.plot(Errors_first)
# plt.show()