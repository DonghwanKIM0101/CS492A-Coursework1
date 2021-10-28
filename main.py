import scipy.io
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

def solve_eig(S):
    eig_val, eig_vec = np.linalg.eigh(S)

    sorted_eig_val = np.array(list(reversed(eig_val)))
    sorted_eig_vec = np.array(list(reversed(eig_vec)))

    positive_map = sorted_eig_val > 0
    positive_sorted_eig_val = sorted_eig_val[positive_map]
    positive_sorted_eig_vec = sorted_eig_vec[positive_map]

    return positive_sorted_eig_val, positive_sorted_eig_vec


start_time = time.time()

mat = scipy.io.loadmat('face.mat')
data_flatten = mat['X']
data = data_flatten.reshape((46,56,-1))
data = np.transpose(data, (1,0,2))

mean_image = np.uint8(np.average(data,2))
mean_flatten = np.average(data_flatten,1)

A = data_flatten - np.repeat(mean_flatten.reshape(-1,1), repeats=520, axis=1)
S = np.matmul(A, A.transpose()) / 520
S_low = np.matmul(A.transpose(), A) / 520

# positive_sorted_eig_val, positive_sorted_eig_vec = solve_eig(S)
# positive_sorted_eig_val_low, positive_sorted_eig_vec_low = solve_eig(S_low)

positive_sorted_eig_val, positive_sorted_eig_vec = solve_eig(S_low)

print(positive_sorted_eig_val.shape)

print("elasped time is ", time.time() - start_time)

# print(positive_sorted_eig_val[:200] - positive_sorted_eig_val_low[:200])

# plt.plot(positive_sorted_eig_val)
# plt.show()


M = 30
index = 0 # image to reconstruct
# phi = A[:,index]
phi = data_flatten[:,index] - mean_flatten

face_recon = mean_flatten

for i in range(M):
    u = positive_sorted_eig_vec[i]
    u = np.matmul(A, u)
    u /= np.linalg.norm(u)

    a = np.matmul(phi.transpose(), u)

    face_recon += a * u

    # cv2.imshow("test%d"%i, np.uint8((face_recon).reshape((46,56)).transpose()))

print("M is %d, reconstruction error is %f"%(M ,np.linalg.norm(face_recon - data_flatten[:,index])))

face_recon = face_recon.reshape((46,56))
face_recon = face_recon.transpose()

cv2.imshow("mean face", mean_image)
cv2.imshow("original", data[:,:,index])

plt.imshow(face_recon, cmap='gist_gray')
plt.show()