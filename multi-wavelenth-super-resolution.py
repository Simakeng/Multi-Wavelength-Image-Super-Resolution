import cv2
import numpy as np

b_scaled, _, _ = cv2.split(cv2.imread("temp/b_scaled.png"))
b_scaled = b_scaled.astype(np.float32)
b_scaled = cv2.dct(b_scaled) * 8
# r = np.zeros((1080,1920))
r, _, _ = cv2.split(cv2.imread("temp/input_r.png"))
r = r.astype(np.float32)
r = cv2.dct(r)
r[0:b_scaled.shape[0],0:b_scaled.shape[1]] = b_scaled
b_restored = cv2.idct(r)

cv2.imwrite("temp/b_restored_idct_true.png", b_restored.astype(np.uint8))