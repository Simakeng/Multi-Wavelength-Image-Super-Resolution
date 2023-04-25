import cv2
import numpy as np

b_scaled, _, _ = cv2.split(cv2.imread("temp/b_scaled.png"))
b_scaled = b_scaled.astype(np.float32)
b_scaled = cv2.dct(b_scaled) * 8
# r = np.zeros((1080,1920))
r, _, _ = cv2.split(cv2.imread("temp/input_r.png"))
r = r.astype(np.float32)
r = cv2.dct(r)
r[0:b_scaled.shape[0], 0:b_scaled.shape[1]] = b_scaled
b_restored = cv2.idct(r)
cv2.imwrite("temp/b_restored_idct_true.png", b_restored.astype(np.uint8))

grad_width = 48
split_h, split_w = b_scaled.shape
res_h, res_w = r.shape
factor_y = np.tile(np.concatenate((np.zeros(split_h - grad_width - 1),
                                   np.arange(0, 1, 1/grad_width),
                                   np.ones(res_h - split_h + 1))),
                   reps=(res_w, 1))
factor_x = np.tile(np.concatenate((np.zeros(split_w - grad_width - 1),
                                   np.arange(0, 1, 1/grad_width),
                                   np.ones(res_w - split_w + 1))),
                   reps=(res_h, 1))
factor_y = 1.0 - factor_y
factor_x = 1.0 - factor_x
factor = factor_y.T * factor_x
cv2.imwrite("temp/factor.png",factor * 255)

r, _, _ = cv2.split(cv2.imread("temp/input_r.png"))
r = r.astype(np.float32)
r = cv2.dct(r)

b_scaled, _, _ = cv2.split(cv2.imread("temp/b_scaled.png"))
b_scaled = b_scaled.astype(np.float32)
b_scaled = cv2.dct(b_scaled) * 8

b = np.zeros(r.shape)
b[0:split_h,0:split_w] = b_scaled

result_dct = b * factor + r * (1 - factor)
result = cv2.idct(result_dct)
cv2.imwrite("temp/b_restored_idct.png", result.astype(np.uint8))

gt, _, _ = cv2.split(cv2.imread("temp/input_b.png"))


a = (abs(gt - result))

import matplotlib.pyplot as plt
plt.imshow(a)
plt.show()