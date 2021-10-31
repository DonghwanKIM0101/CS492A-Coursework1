import scipy.io
import scipy.linalg
import numpy as np
import matplotlib.pyplot as plt

from train_test_split import split
from training_set_split import split_training_set

M = 100
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
    sum = 0
    #print(data.shape[1])
    for i in range(0,data.shape[1]):
        #print(i)
        diff = data[:,i].reshape(-1,1) - mean.reshape(-1,1)
        #print(diff.shape)
        #print(np.matmul(diff, diff.transpose()).shape)
        sum = sum + np.matmul(diff, diff.transpose())
    return sum #, mean

mat = scipy.io.loadmat('face.mat')
data_train, label_train , data_test, label_test = split(mat, TEST_NUMBER)
data = data_train.reshape((WIDTH,HEIGHT,-1))
data = np.transpose(data, (1,0,2))

mean_image = np.uint8(np.average(data,2))
mean_flatten = np.average(data_train,1)

plt.imshow(mean_image, cmap = 'gist_gray')

#within-class scatter matrix
sw = 0
#cnt = 0
for i in range(0, data_train.shape[1]):
    if(i%8 == 0):
        sw = sw + scatter(data_train[:, i:i+8])
        #cnt = cnt + 1

#between-class scatter matrix
sb = 0
for i in range(0, data_train.shape[1]):
    if(i%8 == 0):
        d = data_train[:, i:i+8]
        mean_diff = classMean(data_train[:, i:i+8]).reshape(-1,1) - mean_flatten.reshape(-1,1)
        #print(d.shape[1])
        sb = sb + d.shape[1]*np.matmul(mean_diff, mean_diff.transpose())

print("sw shape : ", sw.shape, "\nsb shape : ", sb.shape)
print("sw rank : ", np.linalg.matrix_rank(sw), "\nsb rank : ", np.linalg.matrix_rank(sb))
print("is sw singular? :", np.linalg.det(sw) == 0)

st = sb + sw
# st = 0
# mean = np.average(data_train, 1)
# for i in range(0, data_train.shape[1]):
#     diff = data_train[:, i].reshape(-1,1) - mean.reshape(-1,1)
#     st = st + np.matmul(diff, diff.transpose())

#Perform PCA to get Wpca with Mpca = 364 (=416-52)
eig_val_st, eig_vec_st = solve_eig(st)
Mpca = 364
Wpca = eig_vec_st[:, :Mpca]

#reduced sw and sb by the PCA
sw_pca = np.matmul(np.matmul(Wpca.transpose(), sw), Wpca)
sb_pca = np.matmul(np.matmul(Wpca.transpose(), sb), Wpca)

eig_val_LDA, eig_vec_LDA = solve_eig(np.matmul(np.linalg.inv(sw_pca), sb_pca))
Wlda = eig_vec_LDA[:, :51]

#Wopt
Wopt = np.matmul(Wpca, Wlda)

#First fisher faces
plt.imshow(Wopt[:,1].reshape((WIDTH,HEIGHT)).transpose(), cmap = 'gist_gray')

plt.plot(eig_val_LDA)
plt.show()

#face reconstruction
phi = data_test[:,INDEX] - mean_flatten

weight = np.matmul(phi.reshape(1,-1), Wopt)

face_recon = mean_flatten + np.matmul(Wopt, weight.transpose()).squeeze()

# print("M is %d, reconstruction error is %f"%(M ,np.linalg.norm(face_recon - data_test[:,INDEX])))

face_recon = face_recon.reshape((WIDTH,HEIGHT))
face_recon = face_recon.transpose()

plt.imshow(mean_image, cmap = 'gist_gray')
plt.imshow(data_test[:,INDEX].reshape((WIDTH,HEIGHT)).transpose(), cmap = 'gist_gray')
plt.imshow(np.uint8(face_recon), cmap = 'gist_gray')
