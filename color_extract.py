import cv2
import numpy as np

img = cv2.imread("input.png")

b, g, r = cv2.split(img)

cv2.imwrite("temp/input_b.png", b)
cv2.imwrite("temp/input_g.png", g)
cv2.imwrite("temp/input_r.png", r)


def blend_scale_by2(img, div=2):
    img = img.astype(int)
    for _ in range(2):
        e = img[0::2]
        o = img[1::2]
        img = e + o
        img = np.transpose(img)
    return img // div


b_scaled = b[::]
for _ in range(3):
    b_scaled = blend_scale_by2(b_scaled, div=1)
b_scaled = b_scaled // 64
b_scaled = b_scaled.astype(np.uint8)
cv2.imwrite("temp/b_scaled.png", b_scaled)


b_restored = cv2.resize(b_scaled,b.shape[::-1], interpolation=cv2.INTER_LINEAR)

cv2.imwrite("temp/b_restored.png", b_restored)
