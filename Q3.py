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

start_time = time.time()
# pid = os.getpid()
# current_process = psutil.Process(pid)
# current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2.**20
# print(f"BEFORE CODE: Current memory KB   : {current_process_memory_usage_as_KB: 9.3f} KB")


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
    #reduced sw and sb by the PCA
    sw_pca = np.matmul(np.matmul(Wpca.transpose(), sw), Wpca)
    sb_pca = np.matmul(np.matmul(Wpca.transpose(), sb), Wpca)

    # eig_val_LDA, eig_vec_LDA = solve_eig(np.divide(sb_pca, sw_pca))
    eig_val_LDA, eig_vec_LDA = solve_eig(np.matmul(np.linalg.inv(sw_pca), sb_pca))
    Wlda = eig_vec_LDA[:, :Mlda]
    
    #Wopt
    Wopt = np.matmul(Wpca, Wlda)

    return Wopt, eig_vec_LDA


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
eig_val_st, eig_vec_st = solve_eig(st)

# face recognition accuracy for various Mpca
Accuracies = []

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

    Accuracies.append(np.sum(label_train[:,np.argmin(error,axis=1)] == label_test) / weight_test.shape[0])

plt.plot(Accuracies)
plt.xlabel("Mpca")
plt.ylabel("Accuracy")
plt.show()


# face recognition accuracy for various Mlda
Accuracies = []

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

    Accuracies.append(np.sum(label_train[:,np.argmin(error,axis=1)] == label_test) / weight_test.shape[0])

plt.plot(Accuracies)
plt.xlabel("Mlda")
plt.ylabel("Accuracy")
plt.show()

# training base model for consufion matrix
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

print("Confusion Matrix Accuracy :", accuracy)

# confusion_matrix for last model
confusion_matrix_result =  confusion_matrix(label_test.squeeze(), label_train[:,np.argmin(error,axis=1)].squeeze())#, normalize='all')
# print(confusion_matrix_result)
# plt.imshow(confusion_matrix_result)
# plt.show()
sns.heatmap(confusion_matrix_result, cmap='Reds')
plt.show()