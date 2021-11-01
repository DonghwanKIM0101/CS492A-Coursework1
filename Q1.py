from numpy.linalg.linalg import eig
import scipy.io
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from train_test_split import split
from tqdm import tqdm

M = 100
INDEX = 8 # image to reconstruct
WIDTH = 46
HEIGHT = 56
TEST_NUMBER = 8 

def solve_eig(S):
    start_time = time.time()
    eig_val, eig_vec = np.linalg.eigh(S)
    print("elasped time is ", time.time() - start_time)
    
    sorted_eig_val = np.flip(eig_val)
    sorted_eig_vec = np.flip(eig_vec, axis=1)

    positive_map = sorted_eig_val > 0
    positive_sorted_eig_val = sorted_eig_val[positive_map]
    positive_sorted_eig_vec = sorted_eig_vec[:,positive_map]

    return positive_sorted_eig_val, positive_sorted_eig_vec


mat = scipy.io.loadmat('face.mat')
data_train, _ , data_test, _ = split(mat, TEST_NUMBER)
data = data_train.reshape(WIDTH,HEIGHT,-1)
data = np.transpose(data, (1,0,2))

mean_image = np.uint8(np.average(data,2))
mean_flatten = np.average(data_train,1)
# cv2.imwrite('mean_image.jpg', np.uint8(mean_image))

A = np.subtract(data_train, mean_flatten.reshape(-1,1))
S = np.matmul(A, A.transpose()) / data_train.shape[1]
S_low = np.matmul(A.transpose(), A) / data_train.shape[1]

eig_val, eig_vec = solve_eig(S)
eig_val_low, eig_vec_low = solve_eig(S_low)

eig_vec_low_ = np.matmul(A, eig_vec_low)
eig_vec_low_ /= np.linalg.norm(eig_vec_low_,axis=0,keepdims=True)

# # Check eigen vectors and eigen values are identical.
# eig_vec_error = 0
# eig_val_error = np.average(np.abs(eig_val[:M] - eig_val_low[:M]))
# for i in range(M):
#     u = eig_vec[:,i]

#     u_low = eig_vec_low[:,i]
#     u_low = np.matmul(A, u_low)
#     u_low /= np.linalg.norm(u_low)

#     eig_vec_error += abs(1 - abs(np.dot(u, u_low) / (np.linalg.norm(u) * np.linalg.norm(u_low))))

# eig_vec_error /= M
# # Plot eigen values.
# print(eig_vec_error)
# print(eig_val_error)
# print(eig_vec.shape)
# print(eig_vec_low.shape)
# plt.plot(eig_val)
# plt.plot(eig_val_low)
# plt.show()


# # Face Reconstruction
# phi = data_test[:,INDEX] - mean_flatten

# weight = np.matmul(phi.reshape(1,-1), eig_vec[:,:M])
# face_recon = mean_flatten + np.matmul(eig_vec[:,:M], weight.transpose()).squeeze()

# weight_low = np.matmul(phi.reshape(1,-1), eig_vec_low_[:,:M])
# face_recon_low = mean_flatten + np.matmul(eig_vec_low_[:,:M], weight_low.transpose()).squeeze()

# face_recon = face_recon.reshape(WIDTH,HEIGHT)
# face_recon = face_recon.transpose()

# face_recon_low = face_recon_low.reshape(WIDTH,HEIGHT)
# face_recon_low = face_recon_low.transpose()

# cv2.imshow("mean face", mean_image)
# cv2.imshow("original", data_test[:,INDEX].reshape(WIDTH,HEIGHT).transpose())

# cv2.imshow("face reconstruction", np.uint8(face_recon))
# cv2.imshow("face reconstruction low", np.uint8(face_recon_low))
# cv2.waitKey(0)


# # Qualitatively face reconstruction
# BASES = [50, 100]
# INDICES = [0, 8]

# for index in INDICES:
#     phi = data_test[:,index] - mean_flatten

#     cv2.imwrite("../Figure/original/original_%d.jpg"%index, data_test[:,index].reshape(WIDTH,HEIGHT).transpose())

#     for m in BASES:
#         print("PCA on bases, %d and index, %d ..."%(m, index))

#         weight_original = np.matmul(phi.reshape(1,-1), eig_vec[:,:m])
#         face_recon_original = mean_flatten + np.matmul(eig_vec[:,:m], weight_original.transpose()).squeeze()

#         weight_low = np.matmul(phi.reshape(1,-1), eig_vec_low_[:,:m])
#         face_recon_low = mean_flatten + np.matmul(eig_vec_low_[:,:m], weight_low.transpose()).squeeze()

#         face_recon_original = face_recon_original.reshape(WIDTH,HEIGHT).transpose()
#         face_recon_low = face_recon_low.reshape(WIDTH,HEIGHT).transpose()

#         cv2.imwrite("../Figure/test/test_%d_original_%d_%d.jpg"%(TEST_NUMBER, m, index), np.uint8(face_recon_original))
#         cv2.imwrite("../Figure/test/test_%d_low_%d_%d.jpg"%(TEST_NUMBER, m, index), np.uint8(face_recon_low))


# Quantitatively face reconstruction
Errors_original = []
Errors_low = []

for m in tqdm(range(eig_val.shape[0])):
    error = 0

    for index in range(data_test.shape[1]):
        phi = data_test[:,index] - mean_flatten

        weight_original = np.matmul(phi.reshape(1,-1), eig_vec[:,:m])
        face_recon_original = mean_flatten + np.matmul(eig_vec[:,:m], weight_original.transpose()).squeeze()

        error += np.linalg.norm(face_recon_original - data_test[:,index])
    
    error /= data_test.shape[1]
    Errors_original.append(error)

for m in tqdm(range(eig_val_low.shape[0])):
    error = 0

    for index in range(data_test.shape[1]):
        phi = data_test[:,index] - mean_flatten

        weight_low = np.matmul(phi.reshape(1,-1), eig_vec_low_[:,:m])
        face_recon_low = mean_flatten + np.matmul(eig_vec_low_[:,:m], weight_low.transpose()).squeeze()

        error += np.linalg.norm(face_recon_low - data_test[:,index])
    
    error /= data_test.shape[1]
    Errors_low.append(error)

plt.plot(Errors_original)
# plt.show()
plt.plot(Errors_low)
plt.show()