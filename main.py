import scipy.io
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

def solve_eig(S):
    eig_val, eig_vec = np.linalg.eigh(S)

    sorted_eig_val = np.flip(eig_val)
    sorted_eig_vec = np.flip(eig_vec, axis=1)

    positive_map = sorted_eig_val > 0
    positive_sorted_eig_val = sorted_eig_val[positive_map]
    positive_sorted_eig_vec = sorted_eig_vec[:,positive_map]

    return positive_sorted_eig_val, positive_sorted_eig_vec


start_time = time.time()

mat = scipy.io.loadmat('face.mat')
data_flatten = mat['X']
data = data_flatten.reshape((46,56,-1))
data = np.transpose(data, (1,0,2))

mean_image = np.uint8(np.average(data,2))
mean_flatten = np.average(data_flatten,1)

A = np.subtract(data_flatten, mean_flatten.reshape((-1,1)))
S = np.matmul(A, A.transpose()) / 520
S_low = np.matmul(A.transpose(), A) / 520

eig_val, eig_vec = solve_eig(S)
eig_val_low, eig_vec_low = solve_eig(S_low)

print("elasped time is ", time.time() - start_time)

# Check eigen vectors and eigen values are identical.
M = 100
print(eig_val[:M] - eig_val_low[:M])

for i in range(M):
    u = eig_vec[:,i]

    u_low = eig_vec_low[:,i]
    u_low = np.matmul(A, u_low)
    u_low /= np.linalg.norm(u_low)

    print(np.dot(u, u_low) / (np.linalg.norm(u) * np.linalg.norm(u_low)))

# Plot eigen values.
print(eig_vec.shape)
print(eig_vec_low.shape)
plt.plot(eig_val)
plt.plot(eig_val_low)
plt.show()


# Face Reconstruction
M = 100
index = 0 # image to reconstruct
phi = A[:,index]

face_recon = mean_flatten

for i in range(M):
    u = eig_vec[:,i]

    # u = eig_vec_low[:,i]
    # u = np.matmul(A, u)
    # u /= np.linalg.norm(u)

    a = np.dot(phi, u)
    face_recon += a * u

    # cv2.imshow("test%d"%i, np.uint8((face_recon).reshape((46,56)).transpose()))

print("M is %d, reconstruction error is %f"%(M ,np.linalg.norm(face_recon - data_flatten[:,index])))

face_recon = face_recon.reshape((46,56))
face_recon = face_recon.transpose()

cv2.imshow("mean face", mean_image)
cv2.imshow("original", data[:,:,index])

plt.imshow(face_recon, cmap='gist_gray')
plt.show()