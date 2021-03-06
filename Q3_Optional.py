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

Mpca = 364
Mlda = 51

T = 10 # the number of random feature subspace
M0 = 100
M1 = Mpca - M0

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

#Perform PCA to get Wpca with Mpca = 364 (=416-52)
eig_val_st, eig_vec_st = solve_eig(st)

feature_subspaces = []
for i in range(T):
    random_idx = np.random.choice(data_train.shape[1] - 1 - M0 , M1, replace=False)
    random_idx = np.sort(random_idx) + M0
    Wpca = np.concatenate((eig_vec_st[:, :M0], eig_vec_st[:, random_idx]), axis=1)
    Wopt, eig_vec_LDA = pca_lda(Wpca, sw, sb)
    feature_subspaces.append(eig_vec_LDA)

# #First fisher faces
# plt.imshow(Wopt[:,1].reshape((WIDTH,HEIGHT)).transpose(), cmap = 'gist_gray')
# plt.show()

# plt.plot(eig_val_LDA)
# plt.show()

# face recognition accuracy
Accuracies_average = []
Accuracies_sum = []
Accuracies_voting = []

A = np.subtract(data_train, mean_flatten.reshape(-1,1))
for m in tqdm(range(Mlda)):
    A_test = np.subtract(data_test, mean_flatten.reshape(-1,1))

    # total_result = np.zeros((data_test.shape[1], data_train.shape[1]))
    # total_result = np.zeros(data_test.shape[1])

    individual_accuracies = np.zeros(T)
    total_error = np.zeros((data_test.shape[1], data_train.shape[1]))
    total_result = np.zeros((T, data_test.shape[1]),dtype='int64')
    voting_result = np.zeros((1,data_test.shape[1]))
    
    for i, eig_vec in enumerate(feature_subspaces):
        Wlda = eig_vec[:,:m]
        Wopt = np.matmul(Wpca,Wlda)

        weight = np.matmul(A.transpose(), Wopt)
        weight_test = np.matmul(A_test.transpose(), Wopt)

        weight_test_expanded = weight_test.reshape(weight_test.shape[0],1,weight_test.shape[1])
        weight_expanded = weight.reshape(1,weight.shape[0],weight.shape[1])
        error = np.subtract(weight_test_expanded, weight_expanded)
        error = np.linalg.norm(error, axis=2)

        total_error += error
        individual_accuracies[i] = np.sum(label_train[:,np.argmin(error,axis=1)] == label_test) / weight_test.shape[0]
        total_result[i,:] = label_train[:,np.argmin(error,axis=1)]

    for i in range(data_test.shape[1]):
        voting_result[0,i] = np.argmax(np.bincount(total_result[:,i]))

    Accuracies_average.append(np.average(individual_accuracies))
    Accuracies_sum.append(np.sum(label_train[:,np.argmin(total_error,axis=1)] == label_test) / weight_test.shape[0])
    Accuracies_voting.append(np.sum(voting_result == label_test) / weight_test.shape[0])

plt.plot(Accuracies_average, label='average')
plt.plot(Accuracies_sum, label = "sum")
plt.plot(Accuracies_voting, label ='majority voting')
plt.legend()
plt.xlabel("Mlda")
plt.ylabel("Accuracy")
plt.show()

# confusion matrix
confusion_matrix_result =  confusion_matrix(label_test.squeeze(), voting_result.squeeze(), normalize='all')
sns.heatmap(confusion_matrix_result, cmap='Reds')
plt.show()

confusion_matrix_result =  confusion_matrix(label_test.squeeze(), label_train[:,np.argmin(total_error,axis=1)].squeeze(), normalize='all')
sns.heatmap(confusion_matrix_result, cmap='Reds')
plt.show()