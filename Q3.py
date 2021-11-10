from random import random
import scipy.io
import scipy.linalg
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil
import os
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix
from train_test_split import split

Mpca = 364 # Maximum Mpca
Mlda = 51 # Maximum Mlda

WIDTH = 46
HEIGHT = 56
TEST_NUMBER = 2
LDA_ELAPSED_TIME = 0
LDA_ONLY_TIME = 0

def solve_eig(S):
    eig_val, eig_vec = np.linalg.eigh(S)

    sorted_eig_val = np.flip(eig_val)
    sorted_eig_vec = np.flip(eig_vec, axis=1)

    positive_map = sorted_eig_val > 0
    positive_sorted_eig_val = sorted_eig_val[positive_map]
    positive_sorted_eig_vec = sorted_eig_vec[:,positive_map]

    return positive_sorted_eig_val, positive_sorted_eig_vec

def classMean(data):
    return np.average(data,1)

#scatter matrix
def scatter(data):
    mean = classMean(data)
    diff = np.subtract(data, mean.reshape(-1,1))
    S = np.matmul(diff, diff.transpose())
    return S


def pca_lda(Wpca, sw, sb):
    global LDA_ELAPSED_TIME
    global LDA_ONLY_TIME

    #reduced sw and sb by the PCA
    sw_pca = np.matmul(np.matmul(Wpca.transpose(), sw), Wpca)
    sb_pca = np.matmul(np.matmul(Wpca.transpose(), sb), Wpca)

    # eig_val_LDA, eig_vec_LDA = solve_eig(np.divide(sb_pca, sw_pca))
    A = np.matmul(np.linalg.inv(sw_pca), sb_pca)
    start_time = time.time()
    eig_val_LDA, eig_vec_LDA = solve_eig(A)
    end_time = time.time()
    LDA_ELAPSED_TIME += end_time - start_time
    LDA_ONLY_TIME += end_time - start_time
    Wlda = eig_vec_LDA[:, :Mlda]
    
    #Wopt
    Wopt = np.matmul(Wpca, Wlda)

    return Wopt, eig_vec_LDA

# Memory
# pid = os.getpid()
# current_process = psutil.Process(pid)
# current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2.**20
# print(f"BEFORE CODE: Current memory KB   : {current_process_memory_usage_as_KB: 9.3f} KB")

mat = scipy.io.loadmat('face.mat')
data_train, label_train , data_test, label_test = split(mat, TEST_NUMBER)
data = data_train.reshape((WIDTH,HEIGHT,-1))
data = np.transpose(data, (1,0,2))

mean_image = np.uint8(np.average(data,2))
mean_flatten = np.average(data_train,1)

# plt.imshow(mean_image, cmap = 'gist_gray')
# plt.show()

#within-class scatter matrix
sw = np.zeros((data_train.shape[0], data_train.shape[0]))
for i in range(0, data_train.shape[1]):
    if(i%8 == 0):
        sw += scatter(data_train[:, i:i+8])

#between-class scatter matrix
sb = np.zeros((data_train.shape[0], data_train.shape[0]))
for i in range(0, data_train.shape[1]):
    if(i%8 == 0):
        d = data_train[:, i:i+8]
        mean_diff = np.subtract(classMean(data_train[:, i:i+8]), mean_flatten).reshape(-1,1)
        sb += d.shape[1]*np.matmul(mean_diff, mean_diff.transpose())

print("sw shape : ", sw.shape, "\nsb shape : ", sb.shape)
print("sw rank : ", np.linalg.matrix_rank(sw), "\nsb rank : ", np.linalg.matrix_rank(sb))
print("is sw singular? :", np.linalg.det(sw) == 0)

st = sb + sw

#Perform PCA
start_time = time.time()
eig_val_st, eig_vec_st = solve_eig(st)
LDA_ELAPSED_TIME += time.time() - start_time



# face recognition accuracy for various Mpca
Accuracies_mpca = []

A = np.subtract(data_train, mean_flatten.reshape(-1,1))

for m in tqdm(range(Mpca)):
    #Perform LDA
    Wpca = eig_vec_st[:, :m]
    Wopt, eig_vec_LDA = pca_lda(Wpca, sw, sb)

    Wlda = eig_vec_LDA[:, :Mlda]
    Wopt = np.matmul(Wpca, Wlda)

    weight = np.matmul(A.transpose(), Wopt)

    A_test = np.subtract(data_test, mean_flatten.reshape(-1,1))
    weight_test = np.matmul(A_test.transpose(), Wopt)

    weight_test_expanded = weight_test.reshape(weight_test.shape[0],1,weight_test.shape[1])
    weight_expanded = weight.reshape(1,weight.shape[0],weight.shape[1])
    error = np.subtract(weight_test_expanded, weight_expanded)
    error = np.linalg.norm(error, axis=2)

    Accuracies_mpca.append(np.sum(label_train[:,np.argmin(error,axis=1)] == label_test) / weight_test.shape[0])

plt.plot(Accuracies_mpca)
plt.xlabel("Mpca")
plt.ylabel("Accuracy")
plt.ylim(0, 0.9)
plt.show()



# face recognition accuracy for various Mlda
Accuracies_mlda = []

A = np.subtract(data_train, mean_flatten.reshape(-1,1))

#Perform LDA
Wpca = eig_vec_st[:, :Mpca]
Wopt, eig_vec_LDA = pca_lda(Wpca, sw, sb)

for m in tqdm(range(Mlda)):
    Wlda = eig_vec_LDA[:, :m]
    Wopt = np.matmul(Wpca, Wlda)

    weight = np.matmul(A.transpose(), Wopt)

    A_test = np.subtract(data_test, mean_flatten.reshape(-1,1))
    weight_test = np.matmul(A_test.transpose(), Wopt)

    weight_test_expanded = weight_test.reshape(weight_test.shape[0],1,weight_test.shape[1])
    weight_expanded = weight.reshape(1,weight.shape[0],weight.shape[1])
    error = np.subtract(weight_test_expanded, weight_expanded)
    error = np.linalg.norm(error, axis=2)

    Accuracies_mlda.append(np.sum(label_train[:,np.argmin(error,axis=1)] == label_test) / weight_test.shape[0])

plt.plot(Accuracies_mlda)
plt.xlabel("Mlda")
plt.ylabel("Accuracy")
plt.ylim(0, 0.9)
plt.show()



# best face recognition accuracy
Acc_best = 0
Mpca_best = 0
Mlda_best = 0

A = np.subtract(data_train, mean_flatten.reshape(-1,1))

for mpca in tqdm(range(Mpca)):
    
    Wpca = eig_vec_st[:, :mpca]
    Wopt, eig_vec_LDA = pca_lda(Wpca, sw, sb)

    for mlda in range(Mlda):
        Wlda = eig_vec_LDA[:, :mlda]
        Wopt = np.matmul(Wpca, Wlda)

        weight = np.matmul(A.transpose(), Wopt)
        A_test = np.subtract(data_test, mean_flatten.reshape(-1,1))
        weight_test = np.matmul(A_test.transpose(), Wopt)

        weight_test_expanded = weight_test.reshape(weight_test.shape[0],1,weight_test.shape[1])
        weight_expanded = weight.reshape(1,weight.shape[0],weight.shape[1])
        error = np.subtract(weight_test_expanded, weight_expanded)
        error = np.linalg.norm(error, axis=2)

        acc = np.sum(label_train[:,np.argmin(error,axis=1)] == label_test) / weight_test.shape[0]
        if acc >= Acc_best:
            Acc_best = acc
            Mpca_best = mpca
            Mlda_best = mlda

print("Acc Best :", Acc_best)
print("Mpca Best :", Mpca_best)
print("Mlda Best :", Mlda_best)


# face recognition accuracy for PCA
# batch PCA
data = data_train
mean_batch = np.average(data,1)
A_batch = np.subtract(data, mean_batch.reshape(-1,1))
S = np.matmul(A_batch, A_batch.transpose()) / data.shape[1]
eig_val_batch, eig_vec_batch = solve_eig(S)

Accuracies_batch = []

#batch PCA
for m in tqdm(range(eig_val_batch.shape[0])):
    weight = np.matmul(A_batch.transpose(), eig_vec_batch[:,:m])

    A_test = np.subtract(data_test, mean_batch.reshape(-1,1))
    weight_test = np.matmul(A_test.transpose(), eig_vec_batch[:,:m])

    count = 0
    for i, test in enumerate(weight_test):
        error = np.subtract(test.reshape(1,-1), weight)
        
        error = np.linalg.norm(error, axis=1)
        count += int(label_train[:,np.argmin(error)] == label_test[:,i])

    Accuracies_batch.append(count / weight_test.shape[0])

print("PCA ACC Best :", np.max(Accuracies_batch))



# training base model for consufion matrix
Mpca = 169
Mlda = 50
Wpca = eig_vec_st[:, :Mpca]
Wopt, eig_vec_LDA = pca_lda(Wpca, sw, sb)
Wlda = eig_vec_LDA[:, :Mlda]
Wopt = np.matmul(Wpca, Wlda)

A = np.subtract(data_train, mean_flatten.reshape(-1,1))

weight = np.matmul(A.transpose(), Wopt)

A_test = np.subtract(data_test, mean_flatten.reshape(-1,1))
weight_test = np.matmul(A_test.transpose(), Wopt)

weight_test_expanded = weight_test.reshape(weight_test.shape[0],1,weight_test.shape[1])
weight_expanded = weight.reshape(1,weight.shape[0],weight.shape[1])
error = np.subtract(weight_test_expanded, weight_expanded)
error = np.linalg.norm(error, axis=2)

accuracy = np.sum(label_train[:,np.argmin(error,axis=1)] == label_test) / weight_test.shape[0]

# pid = os.getpid()
# current_process = psutil.Process(pid)
# current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2.**20
# print(f"AFTER  CODE: Current memory KB   : {current_process_memory_usage_as_KB: 9.3f} KB")

print("PCA-LDA Elapsed Time : ", LDA_ELAPSED_TIME)
print("Only LDA Elapsed Time : ", LDA_ONLY_TIME)
print("Confusion Matrix Accuracy :", accuracy)

# # success & failure cases
# print(label_train[:,np.argmin(error,axis=1)] == label_test)

# # print some examples
# idx = (3, 21, 22)
# for i in idx:
#     data = data_test[:, i-1].reshape((WIDTH, HEIGHT, -1))
#     data = np.transpose(data, (1,0,2))
#     plt.imshow(data, cmap = 'gist_gray')
#     plt.show()

# # confusion_matrix for last model
# confusion_matrix_result =  confusion_matrix(label_test.squeeze(), label_train[:,np.argmin(error,axis=1)].squeeze())#, normalize='all')
# sns.heatmap(confusion_matrix_result, cmap='Reds')
# plt.show()