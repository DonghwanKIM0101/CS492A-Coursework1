import scipy.io
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

start_time = time.time()

mat = scipy.io.loadmat('face.mat')
data_flatten = mat['X']
data = data_flatten.reshape((46,56,-1))
data = np.transpose(data, (1,0,2))

mean_image = np.uint8(np.average(data,2))
mean_flatten = np.average(data_flatten,1)

A = data_flatten - np.repeat(mean_flatten.reshape(-1,1), repeats=520, axis=1)
# S = np.matmul(A, A.transpose()) / 520
S = np.matmul(A.transpose(), A) / 520

eig_val, eig_vec = np.linalg.eigh(S)

# eig_index = sorted(range(len(eig_val)), key=lambda k: eig_val[k], reverse=True)
# sorted_eig_val = eig_val[eig_index]
# sorted_eig_vec = eig_vec[eig_index]
sorted_eig_val = np.array(list(reversed(eig_val)))
sorted_eig_vec = np.array(list(reversed(eig_vec)))

positive_map = sorted_eig_val > 0
positive_sorted_eig_val = sorted_eig_val[positive_map]
positive_sorted_eig_vec = sorted_eig_vec[positive_map]

print(np.sum(positive_map))

print("elasped time is ", time.time() - start_time)

# plt.plot(positive_sorted_eig_val)
# plt.show()


M = 1
index = 0 # image to reconstruct

face_recon = mean_flatten

for i in range(M):
    u = positive_sorted_eig_vec[i]
    u = np.matmul(A, u)
    u /= np.linalg.norm(u)

    a = np.matmul(A[:,index].transpose(), u)

    face_recon += a * u

face_recon += mean_flatten

print("M is %d, reconstruction error is %f"%(M ,np.linalg.norm(face_recon - data_flatten[:,index])))

face_recon = face_recon.reshape((46,56))
face_recon = face_recon.transpose()

cv2.imshow("mean face", mean_image)
cv2.imshow("original", data[:,:,index])

plt.imshow(face_recon, cmap="gist_gray")
plt.show()

# scale = 255 / (np.max(face_recon) - np.min(face_recon))

# face_recon -= np.min(face_recon)
# face_recon *= scale
# face_recon = np.uint8(face_recon)

# cv2.imshow("reconstruction", face_recon)
# cv2.waitKey(0)