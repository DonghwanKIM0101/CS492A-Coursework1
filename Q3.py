import scipy.io
import scipy.linalg
import numpy as np
import cv2
import matplotlib.pyplot as plt

from train_test_split import split

M = 100
Mpca = 364
Mlda = 51

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
Wpca = eig_vec_st[:, :Mpca]

#reduced sw and sb by the PCA
sw_pca = np.matmul(np.matmul(Wpca.transpose(), sw), Wpca)
sb_pca = np.matmul(np.matmul(Wpca.transpose(), sb), Wpca)

eig_val_LDA, eig_vec_LDA = solve_eig(np.divide(sb_pca, sw_pca))
Wlda = eig_vec_LDA[:, :Mlda]

#Wopt
Wopt = np.matmul(Wpca, Wlda)

#First fisher faces
plt.imshow(Wopt[:,1].reshape((WIDTH,HEIGHT)).transpose(), cmap = 'gist_gray')
plt.show()

plt.plot(eig_val_LDA)
plt.show()

#face reconstruction
phi = data_test[:,INDEX] - mean_flatten

weight = np.matmul(phi.reshape(1,-1), Wopt)

face_recon = mean_flatten + np.matmul(Wopt, weight.transpose()).squeeze()

face_recon = face_recon.reshape(WIDTH,HEIGHT)
face_recon = face_recon.transpose()

cv2.imshow("face recon", np.uint8(face_recon))
cv2.waitKey(0)