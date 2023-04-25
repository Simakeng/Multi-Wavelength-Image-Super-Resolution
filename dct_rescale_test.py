import cv2
import numpy as np
import matplotlib.pyplot as plt

r, _, _ = cv2.split(cv2.imread("temp/input_r.png"))
b, _, _ = cv2.split(cv2.imread("temp/input_b.png"))
b_restored, _, _ = cv2.split(cv2.imread("temp/b_restored.png"))
# b_restored = cv2.dct(b_restored.astype(np.float32))
# b_restored_visual = np.log(abs(b_restored) + np.finfo(float).eps)
# print(b_restored_visual.min())
# print(b_restored_visual.max())
# b_restored_visual = (b_restored_visual - b_restored_visual.min()) / (b_restored_visual.max() - b_restored_visual.min())
plt.figure(6,figsize=(16, 9))
plt.subplot(231)
plt.imshow(r, 'gray')
plt.title('Red channel'), plt.xticks([]), plt.yticks([])

plt.subplot(232)
plt.imshow(b, 'gray')
plt.title('Blue channel Ground truth'), plt.xticks([]), plt.yticks([])

plt.subplot(233)
plt.imshow(b_restored, 'gray')
plt.title('Blue channel scaled'), plt.xticks([]), plt.yticks([])


plt.subplot(234)
plt.imshow(np.log(abs(cv2.dct(r.astype(np.float32)))))
plt.title('Red channel frequency domain'), plt.xticks([]), plt.yticks([])

plt.subplot(235)
plt.imshow(np.log(abs(cv2.dct(b.astype(np.float32)))))
plt.title('Blue channel frequency domain'), plt.xticks([]), plt.yticks([])

plt.subplot(236)
plt.imshow(np.log(abs(cv2.dct(b_restored.astype(np.float32)))))
plt.title('Scaled Freqency domain'), plt.xticks([]), plt.yticks([])
plt.show()
